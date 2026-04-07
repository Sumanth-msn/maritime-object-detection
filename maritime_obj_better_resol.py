# ============================================================
# SeaDronesSee PRODUCTION TRAINING PIPELINE (T4 OPTIMIZED)
# - Optimized for small object detection (swimmers, lifesaving)
# - T4 GPU memory optimized (16GB VRAM)
# - 1024px resolution for better small object detection
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
# PATHS
# ============================================================
# ============================================================
# PATHS (ABSOLUTE - REUSING OLD DATASET)
# ============================================================

OLD_PROJECT_ROOT = "/home/ec2-user/Maritime-Obj-Det"
NEW_PROJECT_ROOT = "/home/ec2-user/version2"

RAW_DATASET = f"{OLD_PROJECT_ROOT}/dataset"  # reuse if needed
YOLO_DATASET = f"{OLD_PROJECT_ROOT}/dataset_yolo"  # reuse (IMPORTANT)
TILED_DATASET = f"{NEW_PROJECT_ROOT}/dataset_tiled"  # new tiles here
DATA_YAML = f"{NEW_PROJECT_ROOT}/maritime_dataset.yaml"


# ============================================================
# CLASS MAPPING
# ============================================================
COCO_CATEGORY_MAPPING = {
    1: "swimmer",
    2: "boat",
    3: "jetski",
    4: "lifesaving_appliance",
    5: "buoy",
}

CLASS_NAMES = ["swimmer", "boat", "jetski", "lifesaving_appliance", "buoy"]


# ============================================================
# COCO -> YOLO CONVERSION
# ============================================================
def coco_to_yolo(coco_json, img_dir, out_img_dir, out_lbl_dir):
    """Convert COCO format annotations to YOLO format with proper class mapping."""
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

    for img_id in tqdm(
        coco.getImgIds(), desc=f"Converting {os.path.basename(coco_json)}"
    ):
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

                coco_cat_id = ann["category_id"]
                cls = cat_map.get(coco_cat_id)

                if cls is None:
                    skipped_anns += 1
                    continue

                # Convert to YOLO format
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                bw /= w
                bh /= h

                f.write(f"{cls} {xc} {yc} {bw} {bh}\n")
                has_valid_labels = True

        if not has_valid_labels:
            os.remove(label_path)
            skipped_imgs += 1

    print(f"[INFO] Skipped images (no valid labels): {skipped_imgs}")
    print(f"[INFO] Skipped annotations (ignored categories): {skipped_anns}")


# ============================================================
# IMAGE TILING (1024px with overlap for small objects)
# ============================================================
def tile_image(
    img_path, label_path, out_img_dir, out_lbl_dir, tile_size=1024, overlap=256
):
    """
    Tile large images into 1024x1024 patches with 256px overlap.
    Optimized for small object detection (swimmers, lifesaving appliances).
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

                # Skip if no overlap
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

                # Calculate center and dimensions
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
    """Process all images in a split (train/val) and create 1024px tiles."""
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
            out_lbl,
            tile_size=1024,  # 1024px tiles for better small object detection
            overlap=256,  # 25% overlap to avoid cutting objects
        )
        if not success:
            skipped += 1

    print(f"[INFO] Tiling complete - skipped {skipped} images (no objects)")


# ============================================================
# LABEL VALIDATION
# ============================================================
def validate_labels():
    """Validate all label files and remove corrupt ones."""
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

                    if not (0 <= cls < num_classes):
                        raise ValueError(f"Class {cls} out of range")

                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        raise ValueError(f"Invalid center: x={x}, y={y}")

                    if w <= 0 or h <= 0 or w > 1 or h > 1:
                        raise ValueError(f"Invalid size: w={w}, h={h}")

        except Exception as e:
            corrupt_count += 1
            os.remove(label_file)
            img_path = label_file.replace("/labels/", "/images/").replace(
                ".txt", ".jpg"
            )
            if os.path.exists(img_path):
                os.remove(img_path)

    print(f"[INFO] Validation complete - removed {corrupt_count} corrupt files")


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SEADRONESSEE TRAINING - T4 OPTIMIZED FOR SMALL OBJECTS")
    print("=" * 70)
    print("\nConfiguration:")
    print("  - Tile size: 1024x1024 (better for small objects)")
    print("  - Training size: 1024x1024")
    print("  - Overlap: 256px (avoids cutting objects)")
    print("  - Target: Improve swimmer/lifesaving detection from 80% to 86-88%")
    print("=" * 70)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        print("\n⚠ WARNING: No GPU detected!")

    # ========== STEP 1: COCO TO YOLO ==========
    print("\n" + "=" * 70)
    print("STEP 1: Converting COCO to YOLO format")
    print("=" * 70)

    """coco_to_yolo(
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
    )"""
    # ============================================================
    # SAFETY CHECKS
    # ============================================================
    assert os.path.exists(YOLO_DATASET), (
        f"[ERROR] dataset_yolo not found at {YOLO_DATASET}. "
        "Please verify OLD_PROJECT_ROOT path."
    )
    assert os.path.exists(f"{YOLO_DATASET}/images/train"), (
        "[ERROR] dataset_yolo/images/train not found!"
    )
    assert os.path.exists(f"{YOLO_DATASET}/labels/train"), (
        "[ERROR] dataset_yolo/labels/train not found!"
    )

    print("✓ dataset_yolo found and validated")

    # ========== STEP 2: TILING ==========
    print("\n" + "=" * 70)
    print("STEP 2: Creating 1024x1024 tiles with 256px overlap")
    print("=" * 70)

    if os.path.exists(TILED_DATASET):
        shutil.rmtree(TILED_DATASET)

    tile_split("train")
    tile_split("val")

    # ========== STEP 3: VALIDATION ==========
    print("\n" + "=" * 70)
    print("STEP 3: Validating labels")
    print("=" * 70)

    validate_labels()

    # ========== STEP 4: STATISTICS ==========
    print("\n" + "=" * 70)
    print("STEP 4: Dataset Summary")
    print("=" * 70)

    train_imgs = len(glob.glob(f"{TILED_DATASET}/images/train/*.jpg"))
    val_imgs = len(glob.glob(f"{TILED_DATASET}/images/val/*.jpg"))
    train_lbls = len(glob.glob(f"{TILED_DATASET}/labels/train/*.txt"))
    val_lbls = len(glob.glob(f"{TILED_DATASET}/labels/val/*.txt"))

    print(f"Classes: {len(CLASS_NAMES)}")
    print(f"Names: {CLASS_NAMES}")
    print(f"\nTraining:")
    print(f"  Images: {train_imgs}")
    print(f"  Labels: {train_lbls}")
    print(f"\nValidation:")
    print(f"  Images: {val_imgs}")
    print(f"  Labels: {val_lbls}")

    # ========== STEP 5: YAML CONFIG ==========
    print("\n" + "=" * 70)
    print("STEP 5: Creating YAML configuration")
    print("=" * 70)

    if os.path.exists(DATA_YAML):
        os.remove(DATA_YAML)

    yaml_config = {
        "path": os.path.abspath(TILED_DATASET),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    with open(DATA_YAML, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created {DATA_YAML}")

    # ============================================================
    # DEBUG: PRINT YAML CONTENTS (REMOVE AFTER FIRST RUN)
    # ============================================================
    with open(DATA_YAML) as f:
        print("\n[DEBUG] maritime_dataset.yaml contents:\n")
        print(f.read())

    # ========== STEP 6: TRAINING (T4 OPTIMIZED) ==========
    print("\n" + "=" * 70)
    print("STEP 6: Training YOLOv8-L (T4 Optimized)")
    print("=" * 70)
    print("\nT4-Optimized Configuration:")
    print("  - Model: YOLOv8-L")
    print("  - Image size: 1024x1024 (better for small objects)")
    print("  - Batch size: 4 (T4 safe limit)")
    print("  - Epochs: 25")
    print("  - Multi-scale: 512-1024px (helps small objects)")
    print("  - Mosaic: Enabled (more small objects per batch)")
    print("  - Copy-paste: Enabled (augments small objects)")
    print("\nExpected Training Time: ~20-24 hours on T4")
    print("Expected Improvement:")
    print("  - Large objects: 90% → 92%")
    print("  - Small objects: 80% → 86-88%")
    print("=" * 70)

    model = YOLO("yolov8l.pt")

    results = model.train(
        # Data
        data=DATA_YAML,
        imgsz=1024,  # Optimized for small objects
        # Training duration
        epochs=30,  # Reasonable for convergence
        patience=8,  # Early stopping
        # Batch settings (T4 optimized)
        batch=4,  # Safe for T4 16GB VRAM
        device=0,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,
        cos_lr=True,
        # Small object augmentation
        scale=0.5,  # Multi-scale: 512-1024px
        mosaic=1.0,  # Combine 4 images (more small objects)
        copy_paste=0.1,  # Copy-paste small objects
        # Performance (T4 optimized)
        amp=True,  # CRITICAL: Mixed precision for memory
        workers=4,  # Reduced for T4
        cache=False,  # Don't cache to save RAM
        # Output
        project="outputs",
        name="seadronessee_1024_t4",
        save=True,
        exist_ok=True,
        verbose=True,
    )

    # ========== COMPLETE ==========
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Best model: outputs/seadronessee_1024_t4/weights/best.pt")
    print(f"✓ Last model: outputs/seadronessee_1024_t4/weights/last.pt")
    print(f"✓ Results: outputs/seadronessee_1024_t4/")
    print("\nExpected Performance:")
    print("  - Overall mAP50: 0.92-0.93 (was 0.911)")
    print("  - Large objects: ~92% (was 90%)")
    print("  - Small objects: ~86-88% (was 80%)")
    print("\nTo use the model:")
    print("  model = YOLO('outputs/seadronessee_1024_t4/weights/best.pt')")
    print("  results = model.predict('image.jpg', imgsz=1024)")
    print("=" * 70)
