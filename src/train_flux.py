import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from diffusers import Flux2Pipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import json
import os
import gc


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents

def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 4)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    return latent_image_ids.repeat(batch_size, 1, 1).to(device=device, dtype=dtype)

def main():
    model_id = "diffusers/FLUX.2-dev-bnb-4bit"
    dataset_path = "nature_data/V1 UAV-RSOD_Dataset for Segmentation/1 Images" 
    output_dir = "lora-railway-obstacles-flux2-hpo"
    cache_path = "/models_cache"

    batch_size = 1
    num_epochs = 30 
    gradient_accumulation_steps = 1
    learning_rate = 1.94475632e-05
    lora_alpha = 8
    r = 8
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16

    metadata_path = os.path.join(dataset_path, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for filename in os.listdir(dataset_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                base_name = os.path.splitext(filename)[0]
                txt_path = os.path.join(dataset_path, f"{base_name}.txt")
                caption = open(txt_path, "r", encoding="utf-8").read().strip() if os.path.exists(txt_path) else ""
                f.write(json.dumps({"file_name": filename, "text": caption}) + "\n")

    print("\n--- Phase 1: Pre-computing Latents and Text Embeddings ---")
    pipeline = Flux2Pipeline.from_pretrained(model_id, torch_dtype=weight_dtype, cache_dir=cache_path)
    pipeline.vae.to(device)
    pipeline.text_encoder.to(device)
    pipeline.transformer.to("cpu") 

    dataset = load_dataset("imagefolder", data_dir=dataset_path, split="train")
    target_width = 1920
    target_height = 1088 

    image_transforms = transforms.Compose([
        transforms.Resize((target_height, target_width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), 
    ])

    cached_data = []
    
    for item in tqdm(dataset, desc="Caching HD Data"):
        img = image_transforms(item["image"].convert("RGB")).unsqueeze(0).to(device, dtype=weight_dtype)
        text = item["text"]

        with torch.no_grad():
            latent = pipeline.vae.encode(img).latent_dist.sample()
            shift = getattr(pipeline.vae.config, "shift_factor", 0.0)
            scale = getattr(pipeline.vae.config, "scaling_factor", 1.0)
            
            latent = (latent - shift) * scale
            packed_latent = pack_latents(latent, 1, latent.shape[1], latent.shape[2], latent.shape[3])
            prompt_embeds, text_ids = pipeline.encode_prompt(prompt=text, max_sequence_length=512)
            
            img_ids = prepare_latent_image_ids(1, latent.shape[2], latent.shape[3], device, weight_dtype)

            cached_data.append({
                            "latents": packed_latent[0].cpu(),
                            "prompt_embeds": prompt_embeds[0].cpu(),
                            "text_ids": text_ids[0].cpu() if text_ids.dim() == 3 else text_ids.cpu(),
                            "img_ids": img_ids[0].cpu() if img_ids.dim() == 3 else img_ids.cpu()
                        })

    print("Pre-computation complete. Freeing VRAM...")
    del pipeline.vae
    del pipeline.text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    
    transformer = pipeline.transformer
    transformer.to(device) 
    transformer.requires_grad_(False)
    
    lora_config = LoraConfig(
        r=r,          
        lora_alpha=lora_alpha,   
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"], 
        init_lora_weights="gaussian"
    )
    
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer.enable_gradient_checkpointing()

    train_dataloader = DataLoader(cached_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate, weight_decay=1e-4)

    max_train_steps = num_epochs * (len(train_dataloader) // gradient_accumulation_steps)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=max_train_steps // 10, num_training_steps=max_train_steps)

    print("\n--- Phase 3: Training HD FLUX ---")
    transformer.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            latents = batch["latents"].to(device)
            prompt_embeds = batch["prompt_embeds"].to(device)
            text_ids = batch["text_ids"].to(device)
            img_ids = batch["img_ids"].to(device)

            bsz = latents.shape[0]
            noise = torch.randn_like(latents)
            timesteps = torch.rand((bsz,), device=device)
            sigmas = timesteps.view(-1, 1, 1) 
            
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            target = noise - latents
            timesteps = timesteps * 1000.0

            with torch.autocast('cuda', dtype=weight_dtype):
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    guidance=torch.tensor([4.0] * bsz, device=device), 
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids[0],
                    img_ids=img_ids[0],
                    return_dict=False,
                )[0]
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / gradient_accumulation_steps

            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(train_dataloader):.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs: 
            output_dir_epoch = f"{output_dir}_epoch_{epoch+1}"
            transformer.save_pretrained(output_dir_epoch)

if __name__ == "__main__":
    main()