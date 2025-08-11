from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="signature.yaml", epochs=100, imgsz=640)