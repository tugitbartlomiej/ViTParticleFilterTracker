import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms # Keep torchvision transforms for augmentations
from tqdm.auto import tqdm
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
    Trainer, # Not used in manual loop, but available
    TrainingArguments # Not used in manual loop
)
import shutil # For saving best model
from pathlib import Path # Use pathlib for cleaner path handling
import glob # Needed for finding checkpoints

# --- Dataset Class ---
# (Dataset class remains the same as in the previous version)
class SurgicalToolDataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor, augment=False):
        print("Initializing dataset...")
        self.images_dir = Path(images_dir) # Use Path
        self.processor = processor
        self.augment = augment

        # Define augmentations (can be tuned)
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
                ], p=0.8),
            ])
        else:
            self.transform = None

        # Load annotations
        print(f"Loading annotations from: {annotations_file}")
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        self.categories = coco_data['categories']
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}

        # Create mappings: image_id -> image_info and image_id -> annotations
        self.image_id_to_image = {img['id']: img for img in coco_data['images']}
        self.image_id_to_anns = {}
        print("Mapping annotations to images...")
        for ann in tqdm(coco_data['annotations']):
            img_id = ann['image_id']
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            # Ensure bbox format is float list [x, y, w, h]
            ann['bbox'] = [float(x) for x in ann['bbox']]
            self.image_id_to_anns[img_id].append(ann)

        # Store image IDs ensuring corresponding image info exists
        self.image_ids = []
        all_image_ids_in_json = set(self.image_id_to_image.keys())

        # Add images with annotations first
        for img_id in self.image_id_to_anns.keys():
             if img_id in all_image_ids_in_json:
                  self.image_ids.append(img_id)

        # Add images that might not have annotations (important for background learning)
        images_with_anns_set = set(self.image_ids)
        for img_id in all_image_ids_in_json:
             if img_id not in images_with_anns_set:
                 self.image_ids.append(img_id)
                 if img_id not in self.image_id_to_anns: # Add empty list if truly no annotations
                      self.image_id_to_anns[img_id] = []

        # Remove duplicates just in case
        self.image_ids = sorted(list(set(self.image_ids)))

        print(f"Dataset initialized. Using {len(self.image_ids)} images found in annotations and image list.")


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_id_to_image[image_id]
        # Ensure annotations is a list, even if empty
        annotations = self.image_id_to_anns.get(image_id, [])
        # Ensure annotations format is correct for processor (list of dicts)
        formatted_annotations = []
        for ann in annotations:
             formatted_annotations.append({
                 'bbox': ann['bbox'],
                 'category_id': ann['category_id'],
                 'area': ann.get('area', float(ann['bbox'][2] * ann['bbox'][3])), # Calculate area if missing
                 'iscrowd': ann.get('iscrowd', 0)
             })

        # Construct full image path using Path object
        image_path = self.images_dir / image_info['file_name']
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None # Will be filtered by collate_fn
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        # Apply augmentations TO THE IMAGE ONLY (if enabled)
        if self.augment and self.transform:
            image = self.transform(image)

        # Prepare target in COCO format for the processor
        target = {'image_id': image_id, 'annotations': formatted_annotations}

        # Use processor to prepare image and annotations
        try:
             encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        except Exception as e:
             print(f"Error processing image {image_id} ({image_path}) with processor: {e}")
             print(f"Target annotations passed: {target['annotations']}")
             return None

        # The processor returns a dict with 'pixel_values', 'pixel_mask', and 'labels'.
        pixel_values = encoding["pixel_values"].squeeze(0) # Squeeze batch dim
        pixel_mask = encoding["pixel_mask"].squeeze(0) # Squeeze batch dim

        # Ensure 'labels' exists and has the expected structure
        if not encoding["labels"]:
             labels = {'class_labels': torch.tensor([], dtype=torch.int64), 'boxes': torch.tensor([], dtype=torch.float32)}
        else:
             labels = encoding["labels"][0] # Get the dict for the single image

        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}

# --- Collate Function ---
# (Collate function remains the same)
def collate_fn(batch):
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    try:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
        labels = [item["labels"] for item in batch]
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}
    except Exception as e:
        print(f"Error during collate_fn: {e}")
        return None

# --- Checkpoint Helper Functions ---
def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint file based on epoch number."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"[Checkpoint] Directory not found: {checkpoint_dir}")
        return None

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        print(f"[Checkpoint] No checkpoints found in {checkpoint_dir}")
        return None

    try:
        # Sort by epoch number extracted from filename
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest_checkpoint = checkpoint_files[-1]
        print(f"[Checkpoint] Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    except Exception as e:
        print(f"[Checkpoint] Error parsing checkpoint filenames: {e}")
        return None


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Loads model and optimizer state from a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        print(f"[Checkpoint] Checkpoint file not found: {checkpoint_path}")
        return model, optimizer, 0 # Return original model/optimizer, start from epoch 0

    print(f"[Checkpoint] Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state dict
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif optimizer is None:
            print("[Checkpoint] Warning: Optimizer not provided, skipping loading its state.")
        else:
            print("[Checkpoint] Warning: Optimizer state not found in checkpoint, skipping loading.")


        # Load epoch number
        start_epoch = checkpoint.get('epoch', 0) # Default to 0 if epoch key is missing
        print(f"[Checkpoint] Successfully loaded state from epoch {start_epoch}")
        # Return epoch number to START FROM (the next one)
        return model, optimizer, start_epoch + 1
    except KeyError as e:
        print(f"[Checkpoint] Error loading checkpoint: Missing key {e}. Checkpoint might be corrupt or incomplete.")
        return model, optimizer, 0
    except Exception as e:
        print(f"[Checkpoint] Error loading checkpoint: {e}")
        return model, optimizer, 0


# --- Main Training Function ---
def train(args):
    print("Starting training script...")
    print(f"Arguments passed or defaults used: {args}")

    # --- Sanity check defaults ---
    if not Path(args.images_dir).is_dir():
        print(f"Error: Default or provided images directory not found: {args.images_dir}")
        exit(1)
    if not Path(args.annotations_path).is_file():
        print(f"Error: Default or provided annotations file not found: {args.annotations_path}")
        exit(1)
    # --- End Sanity check ---

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use specific paths from args for outputs
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    best_model_dir = Path(args.best_model_dir)
    final_model_dir = output_dir / "final_model"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = output_dir / "logs"
    logging_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(logging_dir))

    # --- Load Categories ---
    print("Loading category information from annotations file...")
    try:
        with open(args.annotations_path, 'r') as f:
            coco_data = json.load(f)
        categories = coco_data.get('categories', [])
        if not categories:
             raise ValueError("No 'categories' section found or it's empty in annotations file.")
        id2label = {cat['id']: cat['name'] for cat in categories}
        label2id = {v: k for k, v in id2label.items()}
        print(f"Categories loaded: {id2label}")
        if not id2label:
             raise ValueError("No categories found in annotations file.")
    except Exception as e:
        print(f"Error loading categories: {e}")
        return

    # --- Processor ---
    print(f"Loading image processor from checkpoint: {args.model_checkpoint}")
    try:
        processor = DetrImageProcessor.from_pretrained(args.model_checkpoint)
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    # --- Dataset & DataLoader ---
    print("Creating datasets...")
    try:
        full_dataset = SurgicalToolDataset(
            images_dir=args.images_dir,
            annotations_file=args.annotations_path,
            processor=processor,
            augment=args.augment
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    if len(full_dataset) == 0:
        print("Error: Dataset is empty after initialization. Check paths and annotations.")
        return

    # Split dataset
    if args.train_val_split < 1.0:
        train_size = int(args.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if train_size == 0 or val_size == 0:
            print(f"Warning: Dataset size ({len(full_dataset)}) is too small for the split ratio ({args.train_val_split}). Using all data for training.")
            train_dataset = full_dataset
            val_dataset = None
        else:
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
         print("Using entire dataset for training (train_val_split=1.0).")
         train_dataset = full_dataset
         val_dataset = None

    print(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        print("Validation dataset size: 0")

    print("Creating data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # --- Model ---
    print(f"Loading model configuration from checkpoint: {args.model_checkpoint}")
    try:
        config = DetrConfig.from_pretrained(
            args.model_checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    print(f"Setting num_queries in config to: {args.num_queries}")
    config.num_queries = args.num_queries

    print("Loading pre-trained model weights with updated config...")
    try:
        model = DetrForObjectDetection.from_pretrained(
            args.model_checkpoint,
            config=config,
            ignore_mismatched_sizes=True # Keep True for fine-tuning
        ).to(device)
        print(f"Model loaded successfully with num_queries={model.config.num_queries}.")
        # ─── użyj wszystkich GPU, jeśli są widoczne ───────────────────────────
        if torch.cuda.device_count() > 1:
            print(f"[Init] Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return

    # --- Optimizer ---
    print("Setting up optimizer...")
    try:
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    except Exception as e:
        print(f"Error setting up optimizer: {e}")
        return

    # --- Checkpoint Resuming Logic ---
    start_epoch = 0
    if args.resume_training:
        latest_checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint_path:
            # Pass optimizer to load its state as well
            model, optimizer, start_epoch = load_checkpoint(latest_checkpoint_path, model, optimizer, device)
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("[Resume] No checkpoint found to resume from, starting from scratch.")
    else:
        print("[Resume] Not resuming training, starting from scratch.")


    # --- Training Loop ---
    print(f"Starting training loop from epoch {start_epoch}...")
    best_val_loss = float('inf')
    patience_counter = 0
    last_epoch = start_epoch -1 # Initialize correctly based on start_epoch
    global_step_counter = start_epoch * len(train_dataloader) # Approx global step if resuming

    try:
        # Loop starts from start_epoch (0 if not resuming, N if resuming from epoch N-1)
        for epoch in range(start_epoch, args.epochs):
            last_epoch = epoch
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---") # Display 1-based epoch
            model.train()
            train_loss = 0.0
            processed_batches = 0
            progress_bar = tqdm(train_dataloader, desc=f"Training E{epoch+1}") # Use 1-based epoch in desc

            for i, batch in enumerate(progress_bar):
                if batch is None:
                    print(f"Warning: Skipping empty or problematic batch {i+1} in epoch {epoch+1}")
                    continue

                # --- Move batch to device ---
                try:
                    batch_on_device = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch_on_device[k] = v.to(device)
                        elif isinstance(v, list) and k == 'labels':
                            processed_labels = []
                            for label_dict in v:
                                processed_dict = {}
                                for inner_k, inner_v in label_dict.items():
                                    if isinstance(inner_v, torch.Tensor):
                                        processed_dict[inner_k] = inner_v.to(device)
                                    else:
                                        processed_dict[inner_k] = inner_v
                                processed_labels.append(processed_dict)
                            batch_on_device[k] = processed_labels
                        else:
                            batch_on_device[k] = v
                    batch = batch_on_device
                except Exception as e:
                    print(f"\nError moving batch {i+1} to device in epoch {epoch+1}: {e}")
                    continue

                # --- Forward pass ---
                try:
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss_dict = outputs.loss_dict
                except Exception as e:
                    print(f"\nError during forward pass for batch {i+1} in epoch {epoch+1}: {e}")
                    continue

                # Backward pass & Optimize
                try:
                    optimizer.zero_grad()
                    loss.backward()
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                except Exception as e:
                    print(f"\nError during backward pass/optimizer step for batch {i+1} in epoch {epoch+1}: {e}")
                    continue

                # Logging
                batch_loss = loss.item()
                train_loss += batch_loss
                processed_batches += 1
                progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

                writer.add_scalar("Loss/train_batch", batch_loss, global_step_counter)
                for k,v_loss in loss_dict.items():
                    writer.add_scalar(f"Loss_detail/train_{k}", v_loss.item(), global_step_counter)
                global_step_counter += 1 # Increment global step


            if processed_batches > 0:
                 avg_train_loss = train_loss / processed_batches
                 print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
                 writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch) # Log per epoch
            else:
                 print(f"Epoch {epoch+1} - No training batches processed successfully.")
                 avg_train_loss = float('inf')

            # --- Validation Loop ---
            if val_dataloader:
                model.eval()
                val_loss = 0.0
                processed_val_batches = 0
                print("Running validation...")
                progress_bar_val = tqdm(val_dataloader, desc=f"Validation E{epoch+1}")
                last_val_loss_dict = {}

                with torch.no_grad():
                    for i_val, batch_val in enumerate(progress_bar_val):
                        if batch_val is None:
                             print(f"Warning: Skipping empty or problematic validation batch {i_val+1} in epoch {epoch+1}")
                             continue

                        # --- Move batch to device ---
                        try:
                            batch_on_device_val = {}
                            for k, v in batch_val.items():
                                if isinstance(v, torch.Tensor):
                                    batch_on_device_val[k] = v.to(device)
                                elif isinstance(v, list) and k == 'labels':
                                    processed_labels_val = []
                                    for label_dict in v:
                                        processed_dict_val = {}
                                        for inner_k, inner_v in label_dict.items():
                                            if isinstance(inner_v, torch.Tensor):
                                                processed_dict_val[inner_k] = inner_v.to(device)
                                            else:
                                                processed_dict_val[inner_k] = inner_v
                                        processed_labels_val.append(processed_dict_val)
                                    batch_on_device_val[k] = processed_labels_val
                                else:
                                    batch_on_device_val[k] = v
                            batch_val = batch_on_device_val
                        except Exception as e:
                            print(f"\nError moving validation batch {i_val+1} to device in epoch {epoch+1}: {e}")
                            continue

                        # --- Forward pass ---
                        try:
                             outputs_val = model(**batch_val)
                             loss_val = outputs_val.loss
                             loss_dict_val = outputs_val.loss_dict
                             last_val_loss_dict = {k: v.item() for k, v in loss_dict_val.items()}
                        except Exception as e:
                             print(f"\nError during validation forward pass for batch {i_val+1} in epoch {epoch+1}: {e}")
                             continue

                        batch_loss_val = loss_val.item()
                        val_loss += batch_loss_val
                        processed_val_batches += 1
                        progress_bar_val.set_postfix({"loss": f"{batch_loss_val:.4f}"})


                if processed_val_batches > 0:
                    avg_val_loss = val_loss / processed_val_batches
                    print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")
                    writer.add_scalar("Loss/validation_epoch", avg_val_loss, epoch) # Log per epoch
                    for k, v_item in last_val_loss_dict.items():
                        writer.add_scalar(f"Loss_detail/val_{k}_epoch_lastbatch", v_item, epoch)
                else:
                    print(f"Epoch {epoch+1} - No validation batches processed successfully.")
                    avg_val_loss = float('inf')

                # --- Checkpointing & Early Stopping ---
                if val_dataloader and processed_val_batches > 0:
                    # Save checkpoint every N epochs using args.checkpoint_dir
                    # Saving based on 1-based epoch number
                    if (epoch + 1) % args.save_interval == 0:
                        chkpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                        torch.save({
                            'epoch': epoch + 1, # Save 1-based epoch for clarity
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_val_loss,
                        }, chkpt_path)
                        print(f"Checkpoint saved to {chkpt_path}")

                    # Early stopping and saving best model using args.best_model_dir
                    if avg_val_loss < best_val_loss:
                        print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving best model...")
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                        model_to_save.save_pretrained(str(best_model_dir))
                        processor.save_pretrained(str(best_model_dir))
                        print(f"Best model saved in Hugging Face format to: {best_model_dir}")
                    else:
                        patience_counter += 1
                        print(f"Validation loss did not improve for {patience_counter} epoch(s). Best: {best_val_loss:.4f}")
                        if patience_counter >= args.patience:
                            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
                            break

            elif not val_dataloader: # No validation
                 # Saving based on 1-based epoch number
                 if (epoch + 1) % args.save_interval == 0:
                      chkpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                      torch.save({
                          'epoch': epoch + 1, # Save 1-based epoch
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'loss': avg_train_loss,
                      }, chkpt_path)
                      print(f"Checkpoint saved to {chkpt_path} (no validation performed)")


    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # 'last_epoch' variable should hold the epoch number during which interruption occurred
    finally:
        # --- Final Save ---
        print("\nTraining loop finished or interrupted. Saving final model state...")
        try:
             # Save final model to specific final_model_dir
             model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
             model_to_save.save_pretrained(str(final_model_dir))
             processor.save_pretrained(str(final_model_dir))
             print(f"Final model and processor saved to: {final_model_dir}")
        except Exception as e:
            print(f"Error saving final model/processor: {e}")


        # Save final checkpoint only if training actually ran at least one iteration
        if last_epoch >= start_epoch: # Check if loop ran at least once after potential resume
             final_chkpt_path = checkpoint_dir / "final_checkpoint.pth"
             try:
                 # Determine the loss to save based on whether validation ran
                 final_loss = float('inf')
                 if val_dataloader and 'avg_val_loss' in locals() and processed_val_batches > 0 :
                      final_loss = avg_val_loss
                 elif 'avg_train_loss' in locals():
                      final_loss = avg_train_loss

                 torch.save({
                     'epoch': last_epoch + 1, # Save the number of the last completed epoch + 1 (or interrupted epoch + 1)
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': final_loss,
                 }, final_chkpt_path)
                 print(f"Final checkpoint state saved to: {final_chkpt_path}")
             except Exception as e:
                  print(f"Error saving final checkpoint: {e}")


        if val_dataloader and best_val_loss != float('inf'):
             print(f"Best model location: {best_model_dir} (Validation Loss: {best_val_loss:.4f})")
        elif not val_dataloader:
             print("Best model not tracked as validation was disabled or did not run successfully.")

        writer.close()
        print("TensorBoard writer closed.")
        print("Training script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DETR for Surgical Tool Detection")

    # --- Paths using defaults from user's new script ---
    parser.add_argument("--images_dir", type=str,
                        default=r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/images",
                        help="Directory containing training images.")
    parser.add_argument("--annotations_path", type=str,
                        default=r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/augmented_coco_450-14660_20250417_133558.json",
                        help="Path to COCO format annotations JSON file.")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--best_model_dir", type=str,
                        default="./detr_tool_tracking_model_best",
                        help="Directory to save the best model.")
    parser.add_argument("--output_dir", type=str,
                        default="./detr_training_output", # Keep separate dir for logs/final model state
                        help="Main directory for logs and final model state.")
    # --- End of path arguments ---

    # --- Added Resume Argument ---
    # Przykład JAWNEGO ustawienia default=False (nie zmienia zachowania)
    parser.add_argument("--resume_training",
                        action='store_true',
                        default=True,
                        help="Resume training from the latest checkpoint in checkpoint_dir.")
    # --- End Added Argument ---

    # Model & Config
    parser.add_argument("--model_checkpoint", type=str, default="facebook/detr-resnet-50", help="Pre-trained model checkpoint name.")
    parser.add_argument("--num_queries", type=int, default=50, help="Number of object queries (MUST match inference). Recommended: >=10, e.g., 50 or 100.")

    # Dataset & Loader
    parser.add_argument("--train_val_split", type=float, default=0.9, help="Fraction of data to use for training (rest for validation). Set to 1.0 to disable validation.")
    parser.add_argument("--augment", action='store_true', help="Enable basic data augmentation during training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training and validation batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the main model parts.")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for the backbone.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (AdamW).")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Max gradient norm for clipping (set <= 0 to disable).")

    # Saving & Early Stopping
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")

    args = parser.parse_args()

    # --- Sanity Checks ---
    if not Path(args.images_dir).is_dir():
        print(f"Error: Images directory not found: {args.images_dir}")
        exit(1)
    if not Path(args.annotations_path).is_file():
        print(f"Error: Annotations file not found: {args.annotations_path}")
        exit(1)
    if not 0 < args.train_val_split <= 1:
        print(f"Error: train_val_split must be between 0 (exclusive) and 1 (inclusive), got {args.train_val_split}")
        exit(1)
    if args.num_queries <= 0:
        print(f"Error: num_queries must be positive, got {args.num_queries}")
        exit(1)

    # Create output directories (Pathlib handles exists_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.best_model_dir).mkdir(parents=True, exist_ok=True)


    train(args)