import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time

# Constants
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
CLOSE_AREA_MULTIPLIER = 2.0
FRAME_PROCESS_INTERVAL = 3

# Colors in BGR format
BLUE = (255, 0, 0)
TANGERINE = (0, 128, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

SERIAL_PORT = '/dev/ttyACM0'  # Adjust this to match your Arduino's port on the Jetson
BAUD_RATE = 9600

def draw_crosshair(image, x, y, size=20, color=WHITE, thickness=2):
    cv2.line(image, (x - size, y), (x + size, y), color, thickness)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness)

def draw_grid(image, rows=2, cols=3):
    h, w = image.shape[:2]
    dy, dx = h // rows, w // cols

    for y in range(dy, h, dy):
        cv2.line(image, (0, y), (w, y), WHITE, 1)
    for x in range(dx, w, dx):
        cv2.line(image, (x, 0), (x, h), WHITE, 1)

def get_sector(x, y, image_shape, rows=2, cols=3):
    h, w = image_shape[:2]
    dy, dx = h // rows, w // cols
    row = y // dy
    col = x // dx
    return row * cols + col + 1  # 1-indexed

def highlight_sector(image, sector, rows=2, cols=3, color=(0, 255, 0)):
    h, w = image.shape[:2]
    dy, dx = h // rows, w // cols
    row, col = divmod(sector - 1, cols)
    overlay = image.copy()
    cv2.rectangle(overlay, (col * dx, row * dy), ((col + 1) * dx, (row + 1) * dy), color, -1)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

def display_node_status(image, sector, status):
    text = f"{status} node {sector}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, font_thickness = 0.7, 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    text_x = image.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10

    cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, WHITE, font_thickness)

def send_command(ser, sector, action):
    command = f"{sector},{action}\n"
    ser.write(command.encode())

def main():
    # Load the TensorRT engine using Ultralytics YOLO
    model = YOLO('yolov8hs.engine')

    # Open serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    frame_count = 0

    # Class names (assuming two classes: 'open' and 'close')
    class_names = ['open', 'close']
    colors = {'open': BLUE, 'close': TANGERINE}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        if frame_count % FRAME_PROCESS_INTERVAL == 0:
            # Run inference using the TensorRT engine
            results = model(frame, device='cuda')

            # Process detections and send commands to Arduino
            frame_with_boxes = frame.copy()
            draw_grid(frame_with_boxes)

            largest_area = 0
            largest_detection = None

            # Iterate over results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract bounding box coordinates, confidence, and class ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = class_names[class_id]
                    color = colors[class_name]

                    # Filter out detections below the confidence threshold
                    if conf < CONF_THRESHOLD:
                        continue

                    # Draw bounding box
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)

                    # Add label
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame_with_boxes, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    if class_name == 'close':
                        area *= CLOSE_AREA_MULTIPLIER

                    # Update largest detection
                    if area > largest_area:
                        largest_area = area
                        largest_detection = (x1, y1, x2, y2, conf, class_id)

            # If a detection is found
            if largest_detection:
                x1, y1, x2, y2, conf, class_id = largest_detection
                class_name = class_names[class_id]
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2

                # Draw crosshair
                draw_crosshair(frame_with_boxes, centroid_x, centroid_y)

                # Determine sector
                sector = get_sector(centroid_x, centroid_y, frame_with_boxes.shape)

                # Highlight sector
                highlight_color = GREEN if class_name == 'open' else RED
                highlight_sector(frame_with_boxes, sector, color=highlight_color)

                # Display node status
                status = 'open' if class_name == 'open' else 'close'
                display_node_status(frame_with_boxes, sector, status)

                # Send command to Arduino
                send_command(ser, sector, status)

        else:
            # Use the last processed frame if we're skipping this frame
            frame_with_boxes = frame_with_boxes if 'frame_with_boxes' in locals() else frame

        # Display the processed frame
        cv2.imshow('YOLOv8 TensorRT', frame_with_boxes)
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
