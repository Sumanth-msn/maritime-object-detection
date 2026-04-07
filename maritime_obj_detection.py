# ============================================================
# SeaDronesSee PRODUCTION TRAINING PIPELINE (GPU OPTIMIZED)
# - COCO -> YOLO with correct class mapping
# - Image Tiling with safe coordinate handling
# - Label Validation & cleanup
# - YOLOv8-L Training on full dataset
# ============================================================

import os
import cv2
import glob
import yaml
import torch
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO
from ultralytics import YOLO

# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.benchmark = True

# ============================================================
# PATHS - UPDATE THESE FOR YOUR ENVIRONMENT
# ============================================================
RAW_DATASET = "dataset"              # Your main dataset folder
YOLO_DATASET = "dataset_yolo"        # Intermediate YOLO format
TILED_DATASET = "dataset_tiled"      # Final tiled dataset
DATA_YAML = "maritime_dataset.yaml"  # YAML config file (NEW)

# ============================================================
# CLASS MAPPING CONFIGURATION
# Based on SeaDronesSee dataset structure:
# Category 0: ignored (SKIP)
# Category 1: swimmer
# Category 2: boat
# Category 3: jetski
# Category 4: life_saving_appliances
# Category 5: buoy
# ============================================================
COCO_CATEGORY_MAPPING = {
    1: "swimmer",
    2: "boat",
    3: "jetski",
    4: "lifesaving_appliance",
    5: "buoy"
    # Category 0 "ignored" is intentionally not mapped and will be skipped
}

# YOLO class names (indices 0-4)
CLASS_NAMES = ["swimmer", "boat", "jetski", "lifesaving_appliance", "buoy"]

# ============================================================
# COCO -> YOLO CONVERSION (WITH CORRECT MAPPING)
# ============================================================
def coco_to_yolo(coco_json, img_dir, out_img_dir, out_lbl_dir):
    """
    Convert COCO format annotations to YOLO format with proper class mapping.
    Skips category 0 (ignored) and maps categories 1-5 to YOLO classes 0-4.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    coco = COCO(coco_json)
    cat_ids = coco.getCatIds()
   
    # Create mapping from COCO category ID to YOLO class index
    cat_map = {}
    skipped_categories = []
   
    for cid in cat_ids:
        if cid in COCO_CATEGORY_MAPPING:
            class_name = COCO_CATEGORY_MAPPING[cid]
            if class_name in CLASS_NAMES:
                cat_map[cid] = CLASS_NAMES.index(class_name)
            else:
                skipped_categories.append(cid)
        else:
            skipped_categories.append(cid)
   
    print(f"\n[INFO] Processing {os.path.basename(coco_json)}")
    print(f"[INFO] COCO categories found: {cat_ids}")
    print(f"[INFO] Category mapping: {cat_map}")
    if skipped_categories:
        print(f"[INFO] Skipping categories: {skipped_categories} (ignored)")

    skipped_imgs = 0
    skipped_anns = 0

    for img_id in tqdm(coco.getImgIds(), desc=f"Converting {os.path.basename(coco_json)}"):
        img_info = coco.loadImgs(img_id)[0]
        fname = img_info["file_name"]

        src_img = os.path.join(img_dir, fname)
        img = cv2.imread(src_img)
        if img is None:
            skipped_imgs += 1
            continue

        h, w, _ = img.shape
        cv2.imwrite(os.path.join(out_img_dir, fname), img)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        label_path = os.path.join(out_lbl_dir, fname.replace(".jpg", ".txt"))
        has_valid_labels = False

        with open(label_path, "w") as f:
            for ann in anns:
                if ann.get("iscrowd", 0):
                    continue

                x, y, bw, bh = ann["bbox"]
                if bw <= 0 or bh <= 0:
                    continue
               
                # Get YOLO class index
                coco_cat_id = ann["category_id"]
                cls = cat_map.get(coco_cat_id)
               
                if cls is None:
                    skipped_anns += 1
                    continue

                # Convert to YOLO format (normalized coordinates)
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                bw /= w
                bh /= h

                f.write(f"{cls} {xc} {yc} {bw} {bh}\n")
                has_valid_labels = True

        # Remove label file if no valid annotations
        if not has_valid_labels:
            os.remove(label_path)
            skipped_imgs += 1

    print(f"[INFO] Skipped images (no valid labels): {skipped_imgs}")
    print(f"[INFO] Skipped annotations (ignored categories): {skipped_anns}")

# ============================================================
# IMAGE TILING WITH SAFE COORDINATE HANDLING
# ============================================================
def tile_image(img_path, label_path, out_img_dir, out_lbl_dir,
               tile_size=1024, overlap=256):
    """
    Tile large images into smaller patches with overlap.
    Only keeps tiles that contain valid object annotations.
    """
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w, _ = img.shape
    stride = tile_size - overlap

    # Load labels
    labels = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_val = float(parts[0])
                    if 0 <= cls_val < len(CLASS_NAMES):
                        labels.append(list(map(float, parts)))

    base = os.path.splitext(os.path.basename(img_path))[0]
    tiles_created = False

    # Create tiles with sliding window
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
            tile = img[y:y2, x:x2]

            # Skip very small tiles
            if tile.shape[0] < 200 or tile.shape[1] < 200:
                continue

            tile_labels = []

            # Check which objects fall in this tile
            for cls, xc, yc, bw, bh in labels:
                # Convert normalized coords to absolute
                px, py = xc * w, yc * h
                box_w, box_h = bw * w, bh * h
               
                # Calculate box boundaries
                box_x1 = px - box_w / 2
                box_y1 = py - box_h / 2
                box_x2 = px + box_w / 2
                box_y2 = py + box_h / 2
               
                # Calculate overlap between box and tile
                overlap_x1 = max(box_x1, x)
                overlap_y1 = max(box_y1, y)
                overlap_x2 = min(box_x2, x2)
                overlap_y2 = min(box_y2, y2)
               
                overlap_w = overlap_x2 - overlap_x1
                overlap_h = overlap_y2 - overlap_y1
               
                # Skip if no overlap or very small overlap
                if overlap_w <= 0 or overlap_h <= 0:
                    continue
               
                # Calculate overlap ratio
                box_area = box_w * box_h
                overlap_area = overlap_w * overlap_h
                overlap_ratio = overlap_area / box_area if box_area > 0 else 0
               
                # Only keep if at least 30% of object is in tile
                if overlap_ratio < 0.3:
                    continue
               
                # Use clipped box (the part that's actually in the tile)
                clipped_x1 = max(box_x1, x)
                clipped_y1 = max(box_y1, y)
                clipped_x2 = min(box_x2, x2)
                clipped_y2 = min(box_y2, y2)
               
                # Convert to tile-relative coordinates
                rel_x1 = clipped_x1 - x
                rel_y1 = clipped_y1 - y
                rel_x2 = clipped_x2 - x
                rel_y2 = clipped_y2 - y
               
                # Calculate center and dimensions in tile coordinates
                rel_center_x = (rel_x1 + rel_x2) / 2
                rel_center_y = (rel_y1 + rel_y2) / 2
                rel_w = rel_x2 - rel_x1
                rel_h = rel_y2 - rel_y1
               
                # Get tile dimensions
                tile_w = x2 - x
                tile_h = y2 - y
               
                # Normalize to [0, 1]
                nx = rel_center_x / tile_w
                ny = rel_center_y / tile_h
                nw = rel_w / tile_w
                nh = rel_h / tile_h
               
                # Final clamp to valid range
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                nw = max(1e-6, min(1.0, nw))
                nh = max(1e-6, min(1.0, nh))
               
                # Validate before adding
                if 0 <= nx <= 1 and 0 <= ny <= 1 and 0 < nw <= 1 and 0 < nh <= 1:
                    tile_labels.append([int(cls), nx, ny, nw, nh])

            # Only save tiles with annotations
            if not tile_labels:
                continue

            tile_id = f"{base}_{x}_{y}"
            cv2.imwrite(os.path.join(out_img_dir, tile_id + ".jpg"), tile)

            with open(os.path.join(out_lbl_dir, tile_id + ".txt"), "w") as f:
                for label in tile_labels:
                    f.write(" ".join(map(str, label)) + "\n")

            tiles_created = True

    return tiles_created

def tile_split(split):
    """Process all images in a split (train/val) and create tiles."""
    in_img = f"{YOLO_DATASET}/images/{split}"
    in_lbl = f"{YOLO_DATASET}/labels/{split}"
    out_img = f"{TILED_DATASET}/images/{split}"
    out_lbl = f"{TILED_DATASET}/labels/{split}"

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    skipped = 0
    images = [f for f in os.listdir(in_img) if f.endswith(".jpg")]

    for img_name in tqdm(images, desc=f"Tiling {split}"):
        success = tile_image(
            os.path.join(in_img, img_name),
            os.path.join(in_lbl, img_name.replace(".jpg", ".txt")),
            out_img,
            out_lbl
        )
        if not success:
            skipped += 1

    print(f"[INFO] Tiling complete - skipped {skipped} images (no objects)")

# ============================================================
# LABEL VALIDATION AND CLEANUP
# ============================================================
def validate_labels():
    """
    Validate all label files and remove corrupt ones.
    Also removes corresponding images.
    """
    corrupt_count = 0
    num_classes = len(CLASS_NAMES)
   
    for label_file in glob.glob(f"{TILED_DATASET}/labels/**/*.txt", recursive=True):
        try:
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        raise ValueError("Invalid format")
                   
                    cls, x, y, w, h = map(float, parts)
                   
                    # Validate class index
                    if not (0 <= cls < num_classes):
                        raise ValueError(f"Class {cls} out of range")
                   
                    # Validate coordinates
                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        raise ValueError(f"Invalid center: x={x}, y={y}")
                   
                    # Validate dimensions
                    if w <= 0 or h <= 0 or w > 1 or h > 1:
                        raise ValueError(f"Invalid size: w={w}, h={h}")
                       
        except Exception as e:
            corrupt_count += 1
            # Remove corrupt label
            os.remove(label_file)
            # Remove corresponding image
            img_path = label_file.replace("/labels/", "/images/").replace(".txt", ".jpg")
            if os.path.exists(img_path):
                os.remove(img_path)

    print(f"[INFO] Validation complete - removed {corrupt_count} corrupt files")

# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":

    print("=" * 70)
    print("SEADRONESSEE OBJECT DETECTION - PRODUCTION TRAINING PIPELINE")
    print("=" * 70)

    # ========== STEP 1: COCO TO YOLO CONVERSION ==========
    print("\n" + "=" * 70)
    print("STEP 1: Converting COCO annotations to YOLO format")
    print("=" * 70)
   
    coco_to_yolo(
        coco_json=f"{RAW_DATASET}/annotations/instances_train.json",
        img_dir=f"{RAW_DATASET}/images/train",
        out_img_dir=f"{YOLO_DATASET}/images/train",
        out_lbl_dir=f"{YOLO_DATASET}/labels/train"
    )

    coco_to_yolo(
        coco_json=f"{RAW_DATASET}/annotations/instances_val.json",
        img_dir=f"{RAW_DATASET}/images/val",
        out_img_dir=f"{YOLO_DATASET}/images/val",
        out_lbl_dir=f"{YOLO_DATASET}/labels/val"
    )

    # ========== STEP 2: IMAGE TILING ==========
    print("\n" + "=" * 70)
    print("STEP 2: Creating tiled dataset (1024x1024 tiles, 256px overlap)")
    print("=" * 70)
   
    # Clean tiled dataset directory
    if os.path.exists(TILED_DATASET):
        shutil.rmtree(TILED_DATASET)
   
    tile_split("train")
    tile_split("val")

    # ========== STEP 3: LABEL VALIDATION ==========
    print("\n" + "=" * 70)
    print("STEP 3: Validating labels and removing corrupt files")
    print("=" * 70)
   
    validate_labels()

    # ========== STEP 4: DATASET STATISTICS ==========
    print("\n" + "=" * 70)
    print("STEP 4: Dataset Summary")
    print("=" * 70)
   
    train_imgs = len(glob.glob(f"{TILED_DATASET}/images/train/*.jpg"))
    val_imgs = len(glob.glob(f"{TILED_DATASET}/images/val/*.jpg"))
    train_lbls = len(glob.glob(f"{TILED_DATASET}/labels/train/*.txt"))
    val_lbls = len(glob.glob(f"{TILED_DATASET}/labels/val/*.txt"))
   
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print(f"Class names: {CLASS_NAMES}")
    print(f"\nTraining set:")
    print(f"  - Images: {train_imgs}")
    print(f"  - Labels: {train_lbls}")
    print(f"\nValidation set:")
    print(f"  - Images: {val_imgs}")
    print(f"  - Labels: {val_lbls}")
    print(f"\nTotal tiles: {train_imgs + val_imgs}")

    # ========== STEP 5: CREATE YAML CONFIG ==========
    print("\n" + "=" * 70)
    print("STEP 5: Creating YAML configuration")
    print("=" * 70)
   
    # Remove old YAML if exists
    if os.path.exists(DATA_YAML):
        os.remove(DATA_YAML)
        print(f"[INFO] Removed old {DATA_YAML}")
   
    # Create fresh YAML with absolute path
    yaml_config = {
        "path": os.path.abspath(TILED_DATASET),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
   
    with open(DATA_YAML, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
   
    print(f"[INFO] Created {DATA_YAML}")
    print(f"       Path: {yaml_config['path']}")
    print(f"       Classes: {yaml_config['nc']}")
    print(f"       Names: {yaml_config['names']}")

    # ========== STEP 6: TRAINING ==========
    print("\n" + "=" * 70)
    print("STEP 6: Starting YOLOv8-L Training (GPU)")
    print("=" * 70)
    print("\nTraining Configuration:")
    print("  - Model: YOLOv8-L")
    print("  - Image size: 640x640")
    print("  - Epochs: 20")
    print("  - Batch size: 8")
    print("  - Device: GPU (cuda:0)")
    print("  - Optimizer: AdamW")
    print("  - Learning rate: 0.001 -> 0.00001 (cosine)")
    print("")
   
    model = YOLO("yolov8l.pt")

    results = model.train(
        # Data
        data=DATA_YAML,
        imgsz=640,
       
        # Training duration
        epochs=20,
        patience=5,
       
        # Batch settings (optimized for GPU)
        batch=8,           # Adjust based on your GPU memory
        device=0,          # GPU device
       
        # Optimizer settings
        optimizer="AdamW",
        lr0=0.001,         # Initial learning rate
        lrf=0.01,          # Final learning rate (lr0 * lrf)
        warmup_epochs=3,
        weight_decay=0.0005,
        cos_lr=True,       # Cosine learning rate schedule
       
        # Performance
        amp=True,          # Automatic Mixed Precision
        workers=8,
       
        # Output
        project="outputs",
        name="seadronessee_final",
        save=True,
        exist_ok=True,
        verbose=True
    )

    # ========== TRAINING COMPLETE ==========
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest model weights: outputs/seadronessee_final/weights/best.pt")
    print(f"Last model weights: outputs/seadronessee_final/weights/last.pt")
    print(f"\nResults and visualizations: outputs/seadronessee_final/")
    print(f"\nTo use the trained model for inference:")
    print(f"  model = YOLO('outputs/seadronessee_final/weights/best.pt')")
    print(f"  results = model.predict('path/to/image.jpg')")
    print("=" * 70)
