from ultralytics import YOLO

# ============================================================
# RESUME TRAINING FROM LAST CHECKPOINT
# ============================================================

CHECKPOINT = "outputs/seadronessee_final/weights/last.pt"
DATA_YAML = "maritime_dataset.yaml"

model = YOLO(CHECKPOINT)

results = model.train(
    # Dataset
    data=DATA_YAML,
    imgsz=640,

    # Epochs (TOTAL, not additional)
    epochs=50,            # 20 done + up to 30 more
    resume=True,          # 🔑 resume optimizer + scheduler

    # Early stopping
    patience=10,           # 🔑 stops if no val improvement for 5 epochs

    # Hardware
    device=0,
    batch=8,
    amp=True,
    workers=8,

    # Output (same folder)
    project="outputs",
    name="seadronessee_final",
    exist_ok=True,

    verbose=True
)

print("\nTraining resumed successfully.")
print("Best weights:", "outputs/seadronessee_final/weights/best.pt")
print("Last weights:", "outputs/seadronessee_final/weights/last.pt")
