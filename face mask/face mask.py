from ultralytics import YOLO

model_dir = Path.cwd().parent / 'models'

model = YOLO(model_dir / "yolo11n.pt")