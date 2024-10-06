from ultralytics import YOLO

# Load your custom YOLOv8 model
model = YOLO('yolov8hs.pt')  # Replace with the correct path to your model

# Export the model to TensorRT engine format
model.export(format='engine', device='cuda', half=True)  # Use half precision for better performance
