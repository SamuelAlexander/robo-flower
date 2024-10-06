import cv2
import torch
import numpy as np
from ultralytics import YOLO
import colorsys
import argparse

# Constants
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
CLOSE_AREA_MULTIPLIER = 2.0
FRAME_PROCESS_INTERVAL = 1

# Colors (BGR format)
BLUE = (255, 0, 0)
TANGERINE = (0, 128, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

def setup_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def generate_colors(num_classes):
    return [
        [int(255 * c) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0)[::-1]]
        for hue in np.linspace(0, 1, num_classes, endpoint=False)
    ]

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

def highlight_sector(image, sector, rows=2, cols=3, color=(0, 255, 0, 64)):
    h, w = image.shape[:2]
    dy, dx = h // rows, w // cols
    row, col = divmod(sector - 1, cols)
    overlay = image.copy()
    cv2.rectangle(overlay, (col*dx, row*dy), ((col+1)*dx, (row+1)*dy), color, -1)
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

def process_frame(frame, model, colors):
    results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    frame_with_boxes = frame.copy()
    largest_box, largest_area, largest_class = None, 0, None

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf >= CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                
                area = (x2 - x1) * (y2 - y1)
                if class_name == 'close':
                    area *= CLOSE_AREA_MULTIPLIER
                
                if area > largest_area:
                    largest_area = area
                    largest_box = (x1, y1, x2, y2)
                    largest_class = class_name
                
                color = BLUE if class_name == 'open' else TANGERINE if class_name == 'close' else colors[cls]
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                label = f'{class_name} {conf:.2f}'
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(frame_with_boxes, (x1, y1), (x1 + text_size[0], y1 + text_size[1] + 5), color, -1)
                cv2.putText(frame_with_boxes, label, (x1, y1 + text_size[1] + 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

    draw_grid(frame_with_boxes)

    if largest_box:
        x1, y1, x2, y2 = largest_box
        centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        highlight_color = GREEN if largest_class in ['open', None] else RED
        status = "Opening" if largest_class in ['open', None] else "Closing"
        
        draw_crosshair(frame_with_boxes, centroid_x, centroid_y)
        
        sector = get_sector(centroid_x, centroid_y, frame_with_boxes.shape)
        highlight_sector(frame_with_boxes, sector, color=highlight_color)
        display_node_status(frame_with_boxes, sector, status)

    return frame_with_boxes

def main(input_video, output_video):
    device = setup_device()
    print(f"Using device: {device}")

    model = YOLO('yolov8hs.pt')
    colors = generate_colors(len(model.names))

    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_PROCESS_INTERVAL == 0:
            frame_with_boxes = process_frame(frame, model, colors)
        else:
            frame_with_boxes = frame_with_boxes if 'frame_with_boxes' in locals() else frame

        out.write(frame_with_boxes)
        
        # Display progress
        print(f"Processing frame {frame_count}/{total_frames}", end='\r')

        frame_count += 1

    cap.release()
    out.release()
    print("\nProcessing complete. Output saved to", output_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file with YOLO object detection.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_video", help="Path to save the output video file")
    args = parser.parse_args()

    main(args.input_video, args.output_video)
