import os
import torch
import submitit
from pathlib import Path
from diffusers import Flux2Pipeline
from peft import PeftModel

hpo_epoch = 10  

def process_batch(tasks):
    repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
    lora_weight_path = f"lora-railway-obstacles-flux2-hpo_epoch_{hpo_epoch}" 
    cache_path = "/storage/work/raims-oss/models_cache"
    torch_dtype = torch.bfloat16
    target_width = 1920
    target_height = 1088

    pipe = Flux2Pipeline.from_pretrained(
        repo_id, 
        torch_dtype=torch_dtype, 
        cache_dir=cache_path
    )

    if hpo_epoch != 0:
        # Inject LoRA Weights
        print(f"Injecting LoRA weights from {lora_weight_path}...")
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_weight_path)
        pipe.transformer.merge_and_unload()

    pipe.enable_model_cpu_offload()

    for task in tasks:
        txt_file = task["txt_file"]
        output_path = task["output_path"]

        with open(txt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"Generating: {os.path.basename(output_path)}...")

        image = pipe(
            prompt=prompt,
            height=target_height,
            width=target_width,
            generator=torch.Generator(device="cpu").manual_seed(42),
            num_inference_steps=50, 
            guidance_scale=2.0, 
        ).images[0]

        image.save(output_path)

    return f"Batch complete. Processed {len(tasks)} images."


def main():
    prompt_dir = '/nature_data/V1 UAV-RSOD_Dataset for Segmentation/1 Prompts'
    output_dir = f"/images-labels-{hpo_epoch}/"
    os.makedirs(output_dir, exist_ok=True)

    INJECTED_CLASSES = ["Paperbox", "Jerrycan", "Boulder", "Branch", "IronRod", "Barrel", "Person"] 
    WEATHER_CONDITIONS = ["fog", "rain", "sunlight", "evening", "normal"]
    CAMERA_PERSPECTIVES = ["nadir", "low_fpv", "oblique", "normal"]

    folder_path = Path(prompt_dir)
    base_geo_files = list(folder_path.glob("base_geometry_*.txt"))
    
    BASE_IMAGES = []
    for file_path in base_geo_files:
        try:
            image_id = int(file_path.stem.split('_')[-1])
            BASE_IMAGES.append(image_id)
        except ValueError:
            continue
            
    BASE_IMAGES.sort()
    print(f"Detected {len(BASE_IMAGES)} processed base images in {prompt_dir}.")

    all_tasks = []
    skipped_images = 0  # Counter for already generated images
    
    for i in range(316):
        for injected_class in INJECTED_CLASSES:
            clean_class_name = injected_class.replace(' ', '')
            
            # Create a directory for this class
            class_dir = os.path.join(output_dir, clean_class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for weather in WEATHER_CONDITIONS:
                for perspective in CAMERA_PERSPECTIVES:
                    
                    txt_filename = f"{weather}_{perspective}_{clean_class_name}_{i}.txt"
                    txt_file = os.path.join(prompt_dir, txt_filename)
                    output_filename = f"{weather}_{perspective}_{clean_class_name}_{i}.png"
                    output_path = os.path.join(class_dir, output_filename)
                    
                    # Only create a task if the prompt exists AND the output image does NOT exist
                    if os.path.exists(txt_file):
                        if not os.path.exists(output_path):
                            all_tasks.append({
                                "txt_file": txt_file,
                                "output_path": output_path
                            })
                        else:
                            skipped_images += 1

    print(f"Skipped {skipped_images} images that were already generated.")
    print(f"Found {len(all_tasks)} new text prompts to generate.")
    if not all_tasks:
        print("No new tasks found. Exiting.")
        return

    IMAGES_PER_JOB = 60 
    chunks = [all_tasks[i:i + IMAGES_PER_JOB] for i in range(0, len(all_tasks), IMAGES_PER_JOB)]
    print(f"Divided into {len(chunks)} jobs ({IMAGES_PER_JOB} images per job).")
    log_dir = "/slurm_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    executor = submitit.AutoExecutor(folder=log_dir)
    
    executor.update_parameters(
            slurm_partition="",
            slurm_qos="gpu",
            slurm_job_name="Image_Generation",
            slurm_time="04:00:00",       
            slurm_cpus_per_task=14,      
            slurm_mem="64GB",            
            slurm_nodes=1,
            slurm_gpus_per_node=1,      
            slurm_additional_parameters={"gres": "gpu:1"}, 
        )

    print("Submitting array job to cluster...")
    jobs = executor.map_array(process_batch, chunks)
    
if __name__ == "__main__":
    main()