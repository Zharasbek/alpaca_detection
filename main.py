from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")


# Train the model
results = model.train(data="E:\my_files\ML files\YOLO_projects\Alpaca_object_detection\data.yaml", epochs=20)
