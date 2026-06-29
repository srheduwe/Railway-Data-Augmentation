import os
import gc
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from diffusers import Flux2Pipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband

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


class FluxHPO:
    def __init__(self, model_id, dataset_path, cache_path, cached_data_file="cached_latents.pt"):
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.cache_path = cache_path
        self.cached_data_file = cached_data_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.bfloat16
        
        self._prepare_metadata()
        if not os.path.exists(self.cached_data_file):
            self._prepare_and_cache_data()
        else:
            print(f"Loading pre-cached latents from {self.cached_data_file}")
            self.cached_data = torch.load(self.cached_data_file)

    def _prepare_metadata(self):
        metadata_path = os.path.join(self.dataset_path, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w", encoding="utf-8") as f:
                for filename in os.listdir(self.dataset_path):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        base_name = os.path.splitext(filename)[0]
                        txt_path = os.path.join(self.dataset_path, f"{base_name}.txt")
                        caption = open(txt_path, "r", encoding="utf-8").read().strip() if os.path.exists(txt_path) else ""
                        f.write(json.dumps({"file_name": filename, "text": caption}) + "\n")

    def _prepare_and_cache_data(self):
        pipeline = Flux2Pipeline.from_pretrained(self.model_id, torch_dtype=self.weight_dtype, cache_dir=self.cache_path)
        pipeline.vae.to(self.device)
        pipeline.text_encoder.to(self.device)
        pipeline.transformer.to("cpu") 

        dataset = load_dataset("imagefolder", data_dir=self.dataset_path, split="train")
        
        target_width, target_height = 1920, 1088 
        image_transforms = transforms.Compose([
            transforms.Resize((target_height, target_width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), 
        ])

        self.cached_data = []
        for item in tqdm(dataset, desc="Caching HD Data"):
            img = image_transforms(item["image"].convert("RGB")).unsqueeze(0).to(self.device, dtype=self.weight_dtype)
            text = item["text"]

            with torch.no_grad():
                latent = pipeline.vae.encode(img).latent_dist.sample()
                shift = getattr(pipeline.vae.config, "shift_factor", 0.0)
                scale = getattr(pipeline.vae.config, "scaling_factor", 1.0)
                
                latent = (latent - shift) * scale
                packed_latent = pack_latents(latent, 1, latent.shape[1], latent.shape[2], latent.shape[3])
                prompt_embeds, text_ids = pipeline.encode_prompt(prompt=text, max_sequence_length=512)
                img_ids = prepare_latent_image_ids(1, latent.shape[2], latent.shape[3], self.device, self.weight_dtype)

                self.cached_data.append({
                    "latents": packed_latent[0].cpu(),
                    "prompt_embeds": prompt_embeds[0].cpu(),
                    "text_ids": text_ids[0].cpu() if text_ids.dim() == 3 else text_ids.cpu(),
                    "img_ids": img_ids[0].cpu() if img_ids.dim() == 3 else img_ids.cpu()
                })

        print(f"Saving cached data to {self.cached_data_file}...")
        torch.save(self.cached_data, self.cached_data_file)
        
        del pipeline.vae
        del pipeline.text_encoder
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def configspace(self) -> ConfigurationSpace:
        """Define the Hyperparameter Search Space."""
        cs = ConfigurationSpace() 
        
        lr = Float("learning_rate", (1e-6, 5e-5), default=1e-5, log=True)
        r = Categorical("r", [4, 8, 16, 32], default=16)
        alpha = Categorical("lora_alpha", [8, 16, 32, 64], default=16)
        grad_accum = Integer("gradient_accumulation_steps", (1, 8), default=4)
        
        cs.add([lr, r, alpha, grad_accum])
        return cs

    def train(self, config: Configuration, seed: int = 0, budget: int = 30) -> float:
        num_epochs = int(budget)
        print(f"\nEvaluating Config: {dict(config)} | Budget (Epochs): {num_epochs} | Seed: {seed}")
        torch.manual_seed(seed)
        
        pipeline = Flux2Pipeline.from_pretrained(
            self.model_id, 
            torch_dtype=self.weight_dtype, 
            cache_dir=self.cache_path,
            text_encoder=None,
            vae=None
        )
        transformer = pipeline.transformer
        transformer.to(self.device)
        transformer.requires_grad_(False)
        
        lora_config = LoraConfig(
            r=config["r"],          
            lora_alpha=config["lora_alpha"],   
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"], 
            init_lora_weights="gaussian"
        )
        
        transformer = get_peft_model(transformer, lora_config)
        transformer.enable_gradient_checkpointing()

        batch_size = 1
        train_dataloader = DataLoader(self.cached_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

        grad_accum = config["gradient_accumulation_steps"]
        max_train_steps = num_epochs * (len(train_dataloader) // grad_accum)
        lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=max_train_steps // 10, num_training_steps=max_train_steps)

        transformer.train()
        final_epoch_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            
            for step, batch in enumerate(progress_bar):
                latents = batch["latents"].to(self.device)
                prompt_embeds = batch["prompt_embeds"].to(self.device)
                text_ids = batch["text_ids"].to(self.device)
                img_ids = batch["img_ids"].to(self.device)

                bsz = latents.shape[0]
                noise = torch.randn_like(latents)
                timesteps = torch.rand((bsz,), device=self.device)
                sigmas = timesteps.view(-1, 1, 1) 
                
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                target = noise - latents
                timesteps = timesteps * 1000.0

                with torch.autocast('cuda', dtype=self.weight_dtype):
                    model_pred = transformer(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        guidance=torch.tensor([4.0] * bsz, device=self.device), 
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids[0],
                        img_ids=img_ids[0],
                        return_dict=False,
                    )[0]
                    
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss = loss / grad_accum

                loss.backward()
                
                if (step + 1) % grad_accum == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                current_loss = loss.item() * grad_accum
                epoch_loss += current_loss
                progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

            final_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} Completed | Avg Loss: {final_epoch_loss:.4f}")

        del transformer
        del pipeline
        del optimizer
        del lr_scheduler
        gc.collect()
        torch.cuda.empty_cache()
        return final_epoch_loss

if __name__ == "__main__":
    start = time.process_time()
    MODEL_ID = "diffusers/FLUX.2-dev-bnb-4bit"
    DATASET_PATH = "/nature_data/V1 UAV-RSOD_Dataset for Segmentation/1 Images" 
    CACHE_PATH = "/models_cache"
    
    hpo_agent = FluxHPO(model_id=MODEL_ID, dataset_path=DATASET_PATH, cache_path=CACHE_PATH)

    scenario = Scenario(
        hpo_agent.configspace,
        name="Flux2_LoRA_HPO",
        deterministic=True,
        use_default_config=True,
        walltime_limit=30 * 24 * 60 * 60,  # Max walltime: 10 days
        n_trials=30,                      # Total number of configurations to try
        min_budget=1,                     # Min epochs evaluated by Hyperband
        max_budget=15,                    # Max epochs evaluated by Hyperband
        n_workers=1,
    )
    
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget", eta=3)
    
    smac = MFFacade(
        scenario,
        hpo_agent.train,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
    ) 
    
    print("\nStarting SMAC Optimization...")
    incumbent = smac.optimize()
    incumbent_cost = smac.validate(incumbent)
    
    print("-" * 50)
    print(f"Optimization finished in {time.process_time() - start:.2f} seconds.")
    print(f"Best Configuration (Incumbent): {incumbent}")
    print(f"Best Validation Loss: {incumbent_cost}")
    print("-" * 50)