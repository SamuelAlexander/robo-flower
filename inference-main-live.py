import cv2
import torch
import numpy as np
from ultralytics import YOLO
import colorsys
import serial
import time

# Constants
CONF_THRESHOLD = 0.5  # Minimum confidence score for object detection
IOU_THRESHOLD = 0.3  # Intersection over Union threshold for object detection
CLOSE_AREA_MULTIPLIER = 2.0  # We multiply the area of 'close' to compensate for smaller area compared to open
FRAME_PROCESS_INTERVAL = 3  # We only process every nth frame to improve performance

# Colors in BGR format (OpenCV uses BGR instead of RGB)
BLUE = (255, 0, 0)
TANGERINE = (0, 128, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# Serial communication settings
SERIAL_PORT = '/dev/cu.wchusbserial54CE0217321'  # Adjust this to match your Arduino's port
BAUD_RATE = 9600

def setup_device():
    """
    Check if MPS (Metal Performance Shaders) is available for Mac users.
    If not, we use CPU. This helps in using GPU acceleration if available.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def generate_colors(num_classes):
    """
    Generate a unique color for each class the model can detect.
    This helps in visually distinguishing different object classes.
    """
    return [
        [int(255 * c) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0)[::-1]]
        for hue in np.linspace(0, 1, num_classes, endpoint=False)
    ]

def draw_crosshair(image, x, y, size=20, color=WHITE, thickness=2):
    """
    Draw a crosshair on the image. This helps in pinpointing the center of detected objects.
    """
    cv2.line(image, (x - size, y), (x + size, y), color, thickness)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness)

def draw_grid(image, rows=2, cols=3):
    """
    Draw a grid on the image. This divides the image into sectors (nodes).
    """
    h, w = image.shape[:2]
    dy, dx = h // rows, w // cols

    for y in range(dy, h, dy):
        cv2.line(image, (0, y), (w, y), WHITE, 1)
    for x in range(dx, w, dx):
        cv2.line(image, (x, 0), (x, h), WHITE, 1)

def get_sector(x, y, image_shape, rows=2, cols=3):
    """
    Determine which sector (node) a point belongs to.
    We use this to know which node an object is in.
    """
    h, w = image_shape[:2]
    dy, dx = h // rows, w // cols
    row = y // dy
    col = x // dx
    return row * cols + col + 1  # 1-indexed, so we add 1

def highlight_sector(image, sector, rows=2, cols=3, color=(0, 255, 0, 64)):
    """
    Highlight a specific sector (node) on the image.
    This visually shows which node is currently active.
    """
    h, w = image.shape[:2]
    dy, dx = h // rows, w // cols
    row, col = divmod(sector - 1, cols)
    overlay = image.copy()
    cv2.rectangle(overlay, (col*dx, row*dy), ((col+1)*dx, (row+1)*dy), color, -1)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

def display_node_status(image, sector, status):
    """
    Display the status of the current node (opening or closing).
    This provides feedback about what action is being performed.
    """
    text = f"{status} node {sector}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 0.7, 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Position the text in the top right corner
    text_x = image.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    
    # Draw a black background for the text to make it more readable
    cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, WHITE, font_thickness)

def setup_serial():
    """
    Set up the serial connection to the Arduino.
    """
    return serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

def send_command(ser, sector, action):
    """
    Send a command to the Arduino to control an LED.
    """
    command = f"{sector},{action}\n"
    ser.write(command.encode())

def process_frame(frame, model, colors, ser):
    """
    Process a single frame: detect objects, draw bounding boxes, and highlight sectors.
    Now also sends commands to Arduino.
    """
    # Perform object detection
    results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    frame_with_boxes = frame.copy()
    largest_box, largest_area, largest_class = None, 0, None

    # Process each detected object
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf >= CONF_THRESHOLD:
                # Get box coordinates and class
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                
                # Calculate box area, applying multiplier for 'close' objects
                area = (x2 - x1) * (y2 - y1)
                if class_name == 'close':
                    area *= CLOSE_AREA_MULTIPLIER
                
                # Keep track of the largest object
                if area > largest_area:
                    largest_area = area
                    largest_box = (x1, y1, x2, y2)
                    largest_class = class_name
                
                # Determine color based on class
                color = BLUE if class_name == 'open' else TANGERINE if class_name == 'close' else colors[cls]
                
                # Draw bounding box
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                # Add label with class name and confidence
                label = f'{class_name} {conf:.2f}'
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(frame_with_boxes, (x1, y1), (x1 + text_size[0], y1 + text_size[1] + 5), color, -1)
                cv2.putText(frame_with_boxes, label, (x1, y1 + text_size[1] + 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

    # Draw grid on the frame
    draw_grid(frame_with_boxes)

    # Process the largest detected object
    if largest_box:
        x1, y1, x2, y2 = largest_box
        centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Determine highlight color and status based on object class
        highlight_color = GREEN if largest_class in ['open', None] else RED
        status = "open" if largest_class in ['open', None] else "close"
        
        # Draw crosshair at the center of the largest object
        draw_crosshair(frame_with_boxes, centroid_x, centroid_y)
        
        # Highlight the sector containing the largest object
        sector = get_sector(centroid_x, centroid_y, frame_with_boxes.shape)
        highlight_sector(frame_with_boxes, sector, color=highlight_color)
        display_node_status(frame_with_boxes, sector, status)

        # Send command to Arduino
        send_command(ser, sector, status)

    return frame_with_boxes

def main():
    """
    Main function to run the object detection on webcam feed and control Arduino.
    """
    # Setup the device (CPU or GPU)
    device = setup_device()
    print(f"Using device: {device}")

    # Load the YOLO model
    model = YOLO('yolov8hs.pt')
    colors = generate_colors(len(model.names))

    # Setup serial connection to Arduino
    ser = setup_serial()

    # Open the webcam
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        # Process every FRAME_PROCESS_INTERVAL-th frame
        if frame_count % FRAME_PROCESS_INTERVAL == 0:
            frame_with_boxes = process_frame(frame, model, colors, ser)
        else:
            # Use the last processed frame if we're skipping this frame
            frame_with_boxes = frame_with_boxes if 'frame_with_boxes' in locals() else frame

        # Display the processed frame
        cv2.imshow('YOLOv8 Webcam', frame_with_boxes)
        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    main()
