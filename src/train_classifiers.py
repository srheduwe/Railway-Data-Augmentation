import torch
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
from omegaconf import OmegaConf
import time
import hydra
from collections import defaultdict
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from utils import load_data, load_model, load_label_mappings, EarlyStopping

@hydra.main(
        version_base=None,
        config_path="~/open-source-project/config",
        config_name="default_config.yaml",
)
def main(config):
    # 0. Set hyperparameters
    # -------------------------------------------------------------------------
    print("── 0. Set hyperparameters ───────────────────────")

    num_epochs = config.training.num_epochs
    learning_rate = config.model.learning_rate
    weight_decay = config.training.weight_decay
    num_workers = config.data.num_workers
    batch_size = config.model.batch_size
    checkpoint_path = Path(config.training.ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Print hyperparameter configuration
    print(f"  Device: {device}")
    print("  Hyperparameter configuration:")
    print(OmegaConf.to_yaml(config))

    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("── 1. Load Dataset ──────────────────────────────")
    
    # Load index to class label mapping
    label2idx, _ = load_label_mappings(Path(config.data.json_path))
    num_classes = len(label2idx)
    print("  Classes: ", num_classes, label2idx)

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_data(config)
    
    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 2. Model
    # -------------------------------------------------------------------------
    print("── 2. Load Model ────────────────────────────────")

    model = load_model(config).to(device)

    # Print model summary
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params : {total:,}\n")

    # 3. Train
    # -------------------------------------------------------------------------
    print("── 3. Train Model ────────────────────────────────")
    
    # Timestamp for saving checkpoints and history
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Fine-tune the model using Hugging Face Trainer in case of ViT, otherwise use custom training loop
    if config.model.name == 'vit':
        vit_path = checkpoint_path / Path(f'{config.model.name}-{config.data.name}-{timestr}')
        vit_path.mkdir(parents=True, exist_ok=True)

        # Define training arguments
        train_args = TrainingArguments(
            output_dir = str(vit_path),
            optim="adamw_torch",
            save_total_limit=1,
            save_strategy="best",
            eval_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=config.model.learning_rate,
            per_device_train_batch_size=config.model.batch_size,
            per_device_eval_batch_size=config.model.batch_size,
            num_train_epochs=config.training.num_epochs,
            weight_decay=config.training.weight_decay,
            dataloader_num_workers=config.data.num_workers,
            load_best_model_at_end=True,
            metric_for_best_model=config.model.metric_for_best_model,
            greater_is_better=config.model.greater_is_better,
            warmup_steps=config.model.warmup_steps,
            lr_scheduler_type=config.model.lr_scheduler_type,
        )

        # Initialize Hugging Face Trainer
        trainer = Trainer(
            model=model.model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=model.collate_fn,
            compute_metrics=model.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=config.training.patience,
                early_stopping_threshold=config.training.delta
            )]
        )
        
        # ViT fine-tuning
        trainer.train()
        print("  Training complete")

        # 4. Evaluate
        # -------------------------------------------------------------------------
        print("── 4. Evaluate Model ────────────────────────────────")
        
        # ViT evaluation on test dataset
        outputs = trainer.predict(test_dataset)
        print(outputs.metrics)
        trainer.save_metrics('test', outputs.metrics)
    else:
        # Select loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
            )
        
        # Initialize history dict
        history = defaultdict(list)

        # Initialize early stopping
        ckpt_path =  checkpoint_path / Path(f'{config.model.name}-{config.data.name}-{timestr}.pt')
        early_stopping = EarlyStopping(patience=config.training.patience, delta=config.training.delta, verbose=True, ckpt_path=ckpt_path)

        for epoch in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            correct = 0
            history["epoch"].append(epoch)

            # Training phase
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                img, target = images.to(device), labels.to(device)
                logits = model(img)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)
            history["train_loss"].append(train_loss)

            # Validation phase
            model.eval()
            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    img, target = images.to(device), labels.to(device)
                    logits = model(img)
                    loss = criterion(logits, target)
                    val_loss += loss.item()

                    pred = torch.argmax(logits, dim=1)
                    correct += (pred == target).sum().item()

            # Average validation loss
            val_loss /= len(val_loader.dataset)
            val_acc = correct / len(val_loader.dataset)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"  Epoch {epoch} - train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, val acc: {val_acc:.6f}")
            
            # Check early stopping condition
            early_stopping.check_early_stop(val_loss, model)
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch}")
                break

        print("  Training complete")

        # 4. Evaluate
        # -------------------------------------------------------------------------
        print("── 4. Evaluate Model ────────────────────────────────")

        # Load best model
        model = torch.load(ckpt_path, weights_only=False)
        print(f"  Best performing model checkpoint loaded from {ckpt_path}")

        # Evaluate on test dataset
        model.eval()
        test_loss = 0.0
        correct = 0
        for img, target in test_loader:
            img, target = img.to(device), target.to(device)
            logits = model(img)
            loss = criterion(logits, target)
            test_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            correct += (pred == target).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"  Test loss: {test_loss:.6f}, test accuracy: {test_acc:.6f}")

        # Save loss history
        hist_path = checkpoint_path / Path(f'{config.model.name}-{config.data.name}-{timestr}-history.pt')
        torch.save(history, hist_path)

    # Save hyperparameter configuration
    config_path = checkpoint_path / Path(f'{config.model.name}-{config.data.name}-{timestr}-config.yaml')
    OmegaConf.save(config=config, f=config_path)

    

if __name__ == "__main__":
    main()