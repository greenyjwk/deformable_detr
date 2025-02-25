import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from transformers import TrainingArguments, Trainer
import evaluate
from tqdm.auto import tqdm

# ============= Dataset Preparation =============
class CustomCocoDataset(Dataset):
    def __init__(self, annotations_file, img_dir, processor):
        """
        Args:
            annotations_file (string): Path to the COCO annotations JSON file
            img_dir (string): Path to the image directory
            processor (DeformableDetrImageProcessor): The image processor
        """
        from pycocotools.coco import COCO
        self.coco = COCO(annotations_file)
        self.img_dir = img_dir
        self.processor = processor
        self.image_ids = list(self.coco.imgs.keys())
        
        # Map COCO category IDs to contiguous IDs
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat2cont = {coco_id: i for i, coco_id in enumerate(self.cat_ids)}
        self.cont2cat = {i: coco_id for i, coco_id in enumerate(self.cat_ids)}
        
        # Get class names
        self.classes = [self.coco.cats[cat_id]['name'] for cat_id in self.cat_ids]
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        img_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.img_dir, img_info['file_name'])
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path).convert('L')
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            # Skip annotations with empty bounding boxes
            if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
                
            # COCO format [x, y, width, height] to [x1, y1, x2, y2]
            x, y, width, height = ann['bbox']
            boxes.append([x, y, x + width, y + height])
            
            # Map COCO category ID to contiguous ID
            cat_id = ann['category_id']
            labels.append(self.cat2cont[cat_id])
            
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
        
        # Convert to tensors
        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.long)
        target["image_id"] = torch.tensor([image_id])
        target["area"] = torch.tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.tensor(iscrowd, dtype=torch.long)
        
        # Apply processing
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        # Remove the batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
            
        return encoding

# Collate function for DataLoader
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    
    labels = []
    for item in batch:
        label = {}
        label["boxes"] = item["boxes"]
        label["labels"] = item["labels"]
        label["image_id"] = item["image_id"]
        label["area"] = item["area"] if "area" in item else None
        label["iscrowd"] = item["iscrowd"] if "iscrowd" in item else None
        labels.append(label)
        
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    
    return batch

# ============= Model Preparation =============
# Load pretrained model and processor
model_name = "SenseTime/deformable-detr"
processor = DeformableDetrImageProcessor.from_pretrained(model_name)
model = DeformableDetrForObjectDetection.from_pretrained(model_name)

# ============= Dataset Loading =============
# Paths to your dataset
train_annotations_file = "/mnt/storage/ji/data_detr_cmb_only/annotations/train.json"
train_img_dir = "/mnt/storage/ji/data_detr_cmb_only/train"
val_annotations_file = "/mnt/storage/ji/data_detr_cmb_only/annotations/val.json"
val_img_dir = "/mnt/storage/ji/data_detr_cmb_only/val"

# Create datasets
train_dataset = CustomCocoDataset(train_annotations_file, train_img_dir, processor)
val_dataset = CustomCocoDataset(val_annotations_file, val_img_dir, processor)

# Print some dataset information
print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of validation examples: {len(val_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Create data loaders
batch_size = 2  # Adjust based on your GPU memory
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ============= Model Adaptation =============
# Prepare category mappings
num_classes = len(train_dataset.classes)
id2label = {i: cls for i, cls in enumerate(train_dataset.classes)}
label2id = {cls: i for i, cls in enumerate(train_dataset.classes)}

# Update model's config with new number of classes
if num_classes != model.config.num_labels:
    print(f"Modifying classification head from {model.config.num_labels} to {num_classes} classes")
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.num_labels = num_classes
    
    # Resize the classification head
    model.class_labels_classifier = torch.nn.Linear(
        in_features=model.class_labels_classifier.in_features,
        out_features=num_classes + 1  # +1 for the "no object" class
    )

# ============= Evaluation Metrics =============
# Create COCO evaluator
try:
    coco_metric = evaluate.load("coco")
except:
    # If Hugging Face evaluate doesn't have coco, use a manual implementation
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    class CocoMetric:
        def __init__(self, val_annotations_file):
            self.coco_gt = COCO(val_annotations_file)
            
        def compute(self, predictions, references):
            # Convert predictions to COCO format
            coco_predictions = []
            for pred in predictions:
                image_id = pred["image_id"]
                category_id = train_dataset.cont2cat[pred["category_id"]]
                bbox = pred["bbox"]  # [x, y, width, height]
                score = pred["score"]
                
                coco_pred = {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(x) for x in bbox],
                    "score": float(score)
                }
                coco_predictions.append(coco_pred)
            
            # Create COCO predictions object
            coco_dt = self.coco_gt.loadRes(coco_predictions)
            
            # Run evaluation
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Return metrics
            metrics = {
                "precision/mAP": coco_eval.stats[0],  # AP @ IoU=0.5:0.95
                "precision/mAP_50": coco_eval.stats[1],  # AP @ IoU=0.5
                "precision/mAP_75": coco_eval.stats[2],  # AP @ IoU=0.75
                "precision/mAP_small": coco_eval.stats[3],  # AP for small objects
                "precision/mAP_medium": coco_eval.stats[4],  # AP for medium objects
                "precision/mAP_large": coco_eval.stats[5],  # AP for large objects
                "recall/AR_max_1": coco_eval.stats[6],  # AR given 1 detection per image
                "recall/AR_max_10": coco_eval.stats[7],  # AR given 10 detections per image
                "recall/AR_max_100": coco_eval.stats[8],  # AR given 100 detections per image
                "recall/AR_small": coco_eval.stats[9],  # AR for small objects
                "recall/AR_medium": coco_eval.stats[10],  # AR for medium objects
                "recall/AR_large": coco_eval.stats[11]  # AR for large objects
            }
            return metrics
    
    coco_metric = CocoMetric(val_annotations_file)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = []
    
    for i, (logit, label) in enumerate(zip(logits, labels)):
        pred_boxes = logit["pred_boxes"].detach().cpu().numpy()
        pred_scores = logit["pred_logits"].softmax(-1)[:, :-1].max(-1)[0].detach().cpu().numpy()
        pred_labels = logit["pred_logits"].softmax(-1)[:, :-1].max(-1)[1].detach().cpu().numpy()
        image_id = label["image_id"].item()
        
        for box, score, label_idx in zip(pred_boxes, pred_scores, pred_labels):
            if score < 0.05:  # Confidence threshold
                continue
                
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            pred = {
                "image_id": image_id,
                "category_id": label_idx.item(),
                "bbox": [x1, y1, width, height],  # Convert to COCO format [x, y, width, height]
                "score": score.item()
            }
            predictions.append(pred)
    
    # Get references in the right format if needed
    # Note: COCO evaluator uses the ground truth from the val_annotations_file
    
    results = coco_metric.compute(predictions=predictions, references=None)
    return results

# ============= Custom Trainer for DETR-like models =============
class DetrTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract inputs
        pixel_values = inputs["pixel_values"]
        pixel_mask = inputs["pixel_mask"] if "pixel_mask" in inputs else None
        labels = [{k: v.to(model.device) for k, v in label.items() if v is not None} 
                 for label in inputs["labels"]]
        
        # Forward pass
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        
        # Return loss and outputs if needed
        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss

# ============= Training Arguments =============
output_dir = "./deformable-detr-finetuned"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,  # Adjust based on your needs
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    save_strategy="steps",
    save_steps=500,  # Save checkpoint every 500 steps
    learning_rate=1e-5,  # Small learning rate for fine-tuning
    weight_decay=1e-4,
    save_total_limit=3,  # Keep only the 3 best checkpoints
    load_best_model_at_end=True,
    report_to="tensorboard",
    logging_dir="./logs",
    logging_steps=100,
    dataloader_num_workers=4,  # Adjust based on your CPU
    gradient_accumulation_steps=4,  # Effective batch size = batch_size * gradient_accumulation_steps
    fp16=True,  # Use mixed precision training (if supported)
    remove_unused_columns=False,  # Important for custom datasets
)

# ============= Training =============
trainer = DetrTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

print("Starting training...")
trainer.train()

# ============= Save the Fine-tuned Model =============
model.save_pretrained(os.path.join(output_dir, "final_model"))
processor.save_pretrained(os.path.join(output_dir, "final_model"))
print(f"Model saved to {os.path.join(output_dir, 'final_model')}")

# ============= Model Inference Example =============
def inference_example(image_path, confidence_threshold=0.5):
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process outputs
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes, 
        threshold=confidence_threshold
    )[0]
    
    # Print results
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(x, 2) for x in box.tolist()]
        print(
            f"Detected {id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    
    return results