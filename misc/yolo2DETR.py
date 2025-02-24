# import json
# import os
# import cv2

# # Paths
# image_dir = "/mnt/storage/ji/brain_mri_valdo_mayo/valdo_resample_ALFA_YOLO_PNG_epd_gt_box_t2s/images/train"
# label_dir = "/mnt/storage/ji/brain_mri_valdo_mayo/valdo_resample_ALFA_YOLO_PNG_epd_gt_box_t2s/labels/train"
# output_json = "/mnt/storage/ji/Deformable-DETR/data/VALDO/annotations/.json"

# # Category Mapping
# categories = [{"id": 1, "name": "cmb"}]
# images, annotations = [], []

# # Convert YOLO labels to COCO format
# annotation_id = 1
# for image_id, filename in enumerate(os.listdir(image_dir)):
#     print(image_id, filename)
#     if not filename.endswith(".png"): continue
#     img_path = os.path.join(image_dir, filename)
#     img = cv2.imread(img_path)
#     height, width = img.shape[:2]
#     images.append({
#         "id": image_id,
#         "file_name": filename,
#         "height": height,
#         "width": width
#     })

#     label_path = os.path.join(label_dir, filename.replace(".png", ".txt"))
#     if not os.path.exists(label_path): continue

#     with open(label_path, "r") as f:
#         for line in f.readlines():
#             class_id, x_center, y_center, w, h = map(float, line.split())
#             x_min = (x_center - w / 2) * width
#             y_min = (y_center - h / 2) * height
#             bbox_w = w * width
#             bbox_h = h * height

#             annotations.append({
#                 "id": annotation_id,
#                 "image_id": image_id,
#                 "category_id": 1,
#                 "bbox": [x_min, y_min, bbox_w, bbox_h],
#                 "area": bbox_w * bbox_h,
#                 "iscrowd": 0
#             })
#             annotation_id += 1

# # Save JSON
# coco_format = {"images": images, "annotations": annotations, "categories": categories}
# with open(output_json, "w") as f:
#     json.dump(coco_format, f, indent=4)

# print("Dataset converted to COCO format!")



import json
import os
import cv2

def convert_yolo_to_coco(image_dir, label_dir, output_json):
    categories = [{"id": 1, "name": "cmb"}]  # Modify category if needed
    images, annotations = [], []
    annotation_id = 1

    for image_id, filename in enumerate(os.listdir(image_dir)):
        print(image_id, filename)
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip corrupted files

        height, width = img.shape[:2]
        images.append({
            "id": image_id,
            "file_name": filename,
            "height": height,
            "width": width
        })

        label_path = os.path.join(label_dir, filename.replace(".png", ".txt").replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, x_center, y_center, w, h = map(float, line.split())
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                bbox_w = w * width
                bbox_h = h * height

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "area": bbox_w * bbox_h,
                    "iscrowd": 0
                })
                annotation_id += 1

    coco_format = {"images": images, "annotations": annotations, "categories": categories}

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO annotations saved to {output_json}")

# Define paths
train_image_dir = "/mnt/storage/ji/brain_mri_valdo_mayo/valdo_resample_ALFA_YOLO_PNG_epd_gt_box_t2s_cmb_slice_only/images/train"
train_label_dir = "/mnt/storage/ji/brain_mri_valdo_mayo/valdo_resample_ALFA_YOLO_PNG_epd_gt_box_t2s_cmb_slice_only/labels/train"
val_image_dir = "/mnt/storage/ji/brain_mri_valdo_mayo/valdo_resample_ALFA_YOLO_PNG_epd_gt_box_t2s_cmb_slice_only/images/val"
val_label_dir = "/mnt/storage/ji/brain_mri_valdo_mayo/valdo_resample_ALFA_YOLO_PNG_epd_gt_box_t2s_cmb_slice_only/labels/val"

output_train_json = "/mnt/storage/ji/data_detr_cmb_only/annotations/train.json"
output_val_json = "/mnt/storage/ji/data_detr_cmb_only/annotations/val.json"

# Convert YOLO to COCO
convert_yolo_to_coco(train_image_dir, train_label_dir, output_train_json)
convert_yolo_to_coco(val_image_dir, val_label_dir, output_val_json)