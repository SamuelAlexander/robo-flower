# Robo-Flower: Real-Time Hand Gesture Control of Kinetic Sculptures using Object Detection
Gesture-controlled kinetic sculpture using YOLOv8. Six robotic flowers move based on hand signs.

## Introduction

Imagine a world where buildings breathe and structures respond to a wave of your hand. Our project, "Real-Time Hand Gesture Control of Kinetic Sculptures using Object Detection," bridges art, architecture, and cutting-edge technology to bring this vision to life.

Using Roboflow for dataset preparation, Ultralytics YOLOv8 for object detection, and ESP32 with Arduino IDE for controlling stepper motors with precise movements, we're creating a small-scale kinetic sculpture that responds to hand gestures. This interactive installation serves as a proof of concept for larger architectural applications, from energy-efficient shape-shifting buildings to citywide interactive art installations.

While we start with 6 nodes powered by stepper motors and controlled by an ESP32 board, the principles explored here could potentially scale to thousands, transforming how we interact with our built environment.

In this project, you'll gain hands-on experience with:

1. Computer vision and machine learning using YOLOv8 object detection
2. Dataset creation and augmentation with Roboflow
3. Model training using Google Colab
4. ESP32 with Arduino programming for real-time control
5. Integration of stepper motors into mechanical design for precise movement
6. Bringing together 'high-level' software and 'low-level' hardware control

By the end, you'll have created your own gesture-controlled kinetic sculpture and gained insights into the future of responsive design.

Let's dive in and start building our interactive kinetic sculpture, step by step. We'll begin by setting up our development environment and then move on to creating our custom hand gesture detection model using Roboflow and YOLOv8.

## Project Overview

Our project consists of two main components:

1. A Python-based computer vision system that detects hand gestures in real-time.
2. An ESP32-controlled kinetic sculpture that responds to these gestures.

The system works as follows:
- A webcam captures live video of hand gestures.
- A YOLO (You Only Look Once) object detection model processes the video stream to identify hand gestures.
- The video frame is divided into 6 sectors, each corresponding to a node of the kinetic sculpture.
- Detected gestures in each sector trigger commands sent to an Arduino.
- The Arduino controls stepper motors to move the corresponding nodes of the sculpture.

## Hardware and Software Requirements

### Hardware:
- Computer with a webcam (We will explore hardware acceleration with Apple Silicon and TensorRT)
- ESP32 (ESP32-S3-DevKitC-1 with custom carrier board)
- 6 stepper motors
- 6 stepper motor drivers (e.g., TMC2208)
- 6 limit switches
- Power supply for the motors (12V 5A)
- The physical structure of your kinetic sculpture

### Software:
- Python 3.7 or later
- OpenCV
- PyTorch
- Ultralytics YOLO
- pyserial
- Arduino IDE

## Setting Up the Environment

Let's start by setting up our development environment:

1. Install Python:
   Download and install the latest version of Python from [python.org](https://www.python.org/).

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv roboflower_env
   source roboflower_env/bin/activate  # On Windows, use `roboflower_env\Scripts\activate`
   ```

3. Install required Python libraries:
   ```
   pip install opencv-python torch ultralytics pyserial
   ```

4. Install Arduino IDE:
   Download and install from [arduino.cc](https://www.arduino.cc/en/software).

5. Clone the YOLOv8 repository (needed only for testing the `yolov8n.pt` base model)
   ```
   git clone https://github.com/ultralytics/yolov8.git
   cd yolov8
   pip install -e .
   ```

## Building the Custom Hand Gesture Detection Model (Roboflow + Google Colab)

We'll create a custom hand gesture detection model. This process is crucial for accurately recognizing the specific gestures that will control our kinetic sculpture. Let's dive into the detailed steps:

### Data Collection

1. **Video Recording:**
   - Use a webcam or smartphone camera to record a 2-3 minute video.
   - Demonstrate both closed-fist and open-palm gestures.
   - Vary your hand positions, angles, and distances from the camera.
   - Include different backgrounds and lighting conditions to improve model robustness.
   - Move your hands at different speeds to capture various motion blur scenarios.
   - Include partial hand visibility scenarios (e.g., hand partially out of frame).
   - If possible, involve multiple people to increase hand diversity in the dataset.
   - Aim for at least 100 examples of each gesture.


### Data Preparation with Roboflow

1. **Roboflow Account Setup:**
   - Go to [Roboflow](https://roboflow.com/) and sign up for an account.
   - Create a new project, naming it something like "Hand Gesture Detection".

2. **Video Upload and Processing:**
   - Upload your recorded video to the project.
   - Use Roboflow's video-to-image conversion tool:
     - Set the frame extraction rate to 1 frame per second.
     - This should give you approximately 120-180 images from a 2-3 minute video.

3. **Image Annotation:**
   - Use Roboflow's annotation tool to draw bounding boxes around each hand gesture.
   - Label each gesture as either "open" for open-palm or "close" for closed-fist.
   - Ensure the bounding box tightly encompasses the hand but includes all visible parts.

4. **Using Annotation Assistance Tools:**
   - Explore Roboflow's auto-annotation features like Grounding DINO:
     - Go to the "Annotate" tab and select "Auto-Annotate".
     - Choose "Grounding DINO" and enter prompts like "palm', "hand" or "fist".
     - Review and adjust the auto-generated annotations for accuracy.

5. **Dataset Augmentation:**
   - In the "Generate" tab, add preprocessing steps:
     - Enable "Flip: Horizontal" to double your dataset size and improve model generalization.
     - Consider adding "Rotate: Between -15° and +15°" for rotation invariance.
     - Add "Brightness: Between -25% and +25%" to account for lighting variations (optional).

6. **Generate and Download Dataset:**
   - Click "Generate" to create your final dataset.
   - In the download options, select "YOLO v8" as the format.
   - Copy the code snippet.

### Model Training with Google Colab

1. **Set Up Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/).
   - Open our hand-signs detector (based on the How to Train YOLOv8 Object Detection on a Custom Dataset notebook from the roboflow/notebooks repository)

2. **Prepare the Training Environment:**
ROBOFLOW_API_KEY

3. **Load and Prepare the Dataset:**
   - Use the Roboflow library to download your dataset:
     ```python
     from roboflow import Roboflow
     rf = Roboflow(api_key="YOUR_API_KEY")
     project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
     dataset = project.version(1).download("yolov8")
     ```

4. **Configure and Start Training:**
   - Load the YOLOv8 model and start training:
     ```python
     from ultralytics import YOLO

     # Load a model
     model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for faster training and inference

     # Train the model
     results = model.train(
       data='/content/datasets/YOUR_DATASET/data.yaml',
       epochs=100,
       imgsz=640,
       plots=True
     )
     ```

5. **Monitor Training Progress:**
   - Colab will display training progress, including loss metrics and performance on the validation set.
   - Training typically takes 1-2 hours, depending on your dataset size and chosen epochs.

6. **Evaluate the Model:**
   - After training, run validation to get performance metrics:
     ```python
     results = model.val()
     ```

7. **Save and Download the Model:**
   - Save the best performing model to your Google Drive:
     ```python
     model.save('/content/drive/MyDrive/hand_gesture_model.pt')
     ```
   - Download the entire training folder from Google Drive, which includes your model, training logs, and performance plots.

### Step 4: Implementing the Custom Model

1. **Locate the Trained Model:**
   - In the downloaded zip from Google Drive, find the `hand_gesture_model.pt` file.

2. **Prepare for Implementation:**
   - Place `hand_gesture_model.pt` in your project directory.
   - Update your Python script to use this custom model:
     ```python
     model = YOLO('path/to/hand_gesture_model.pt')
     ```

3. **Test the Model:**
   - Run a quick inference test to ensure the model is working as expected:
     ```python
     import cv2
     from ultralytics import YOLO

     model = YOLO('path/to/hand_gesture_model.pt')
     img = cv2.imread('test_image.jpg')
     results = model(img)
     ```

By following this comprehensive guide, you've created a custom YOLO model specifically trained to recognize hand gestures for your kinetic sculpture project. This tailored approach ensures higher accuracy and reliability in gesture detection, which is crucial for the seamless control of your sculpture.

In the next stage, we'll integrate this custom model into our real-time hand gesture detection system.

## Building the Computer Vision System

We'll build our Python script in stages, starting with basic inferencing and gradually adding features.

### Stage 1: Basic Inferencing

Let's start with a simple script that captures video from the webcam and performs object detection:

```python
import cv2
from ultralytics import YOLO

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the YOLO model
model = YOLO('yolov8n.pt')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Perform detection
    results = model(frame)
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # Display the resulting frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
```

This script does the following:
1. Initializes the webcam.
2. Loads a pre-trained YOLO model.
3. Continuously captures frames from the webcam.
4. Performs object detection on each frame.
5. Displays the annotated frame with detection results.

Run this script to see basic object detection in action.

### Stage 2: Hand Gesture Detection

Now, let's modify our script to focus on hand gestures. We'll use a custom-trained YOLO model for this purpose. (Note: Training a custom model is beyond the scope of this tutorial, but you can use transfer learning on a pre-trained YOLO model with your own hand gesture dataset.)

```python
import cv2
from ultralytics import YOLO

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the custom YOLO model
model = YOLO('path_to_your_custom_model.pt')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Perform detection
    results = model(frame)
    
    # Process the results
    for r in results:
        for box in r.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get the class name
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Hand Gesture Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
```

This script is similar to the previous one, but it processes the detection results to draw bounding boxes and labels for the detected hand gestures.

### Stage 3: Dividing the Frame into Sectors

Next, we'll divide the frame into sectors and determine which sector contains the detected hand gesture:

```python
import cv2
from ultralytics import YOLO

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the custom YOLO model
model = YOLO('path_to_your_custom_model.pt')

# Define grid dimensions
GRID_ROWS = 2
GRID_COLS = 3

def get_sector(x, y, frame_shape):
    h, w = frame_shape[:2]
    row = int(y / (h / GRID_ROWS))
    col = int(x / (w / GRID_COLS))
    return row * GRID_COLS + col + 1

def draw_grid(frame):
    h, w = frame.shape[:2]
    for i in range(1, GRID_ROWS):
        cv2.line(frame, (0, int(i * h / GRID_ROWS)), (w, int(i * h / GRID_ROWS)), (255, 255, 255), 1)
    for i in range(1, GRID_COLS):
        cv2.line(frame, (int(i * w / GRID_COLS), 0), (int(i * w / GRID_COLS), h), (255, 255, 255), 1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Perform detection
    results = model(frame)
    
    # Draw grid
    draw_grid(frame)
    
    # Process the results
    for r in results:
        for box in r.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get the class name
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Calculate center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Determine the sector
            sector = get_sector(center_x, center_y, frame.shape)
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} - Sector {sector}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Hand Gesture Detection with Sectors', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
```

This script adds functions to divide the frame into a 2x3 grid and determine which sector contains the detected hand gesture.

### Stage 4: Serial Communication with Arduino

Finally, let's add serial communication to send commands to the Arduino based on the detected gestures and their sectors:

```python
import cv2
from ultralytics import YOLO
import serial
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the custom YOLO model
model = YOLO('path_to_your_custom_model.pt')

# Define grid dimensions
GRID_ROWS = 2
GRID_COLS = 3

# Setup serial communication
ser = serial.Serial('usbserial1234', 9600, timeout=1)  # Replace 'usbserial1234' with your Arduino's port
time.sleep(2)  # Wait for the serial connection to initialize

def get_sector(x, y, frame_shape):
    h, w = frame_shape[:2]
    row = int(y / (h / GRID_ROWS))
    col = int(x / (w / GRID_COLS))
    return row * GRID_COLS + col + 1

def draw_grid(frame):
    h, w = frame.shape[:2]
    for i in range(1, GRID_ROWS):
        cv2.line(frame, (0, int(i * h / GRID_ROWS)), (w, int(i * h / GRID_ROWS)), (255, 255, 255), 1)
    for i in range(1, GRID_COLS):
        cv2.line(frame, (int(i * w / GRID_COLS), 0), (int(i * w / GRID_COLS), h), (255, 255, 255), 1)

def send_command(sector, gesture):
    command = f"{sector},{gesture}\n"
    ser.write(command.encode())

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Perform detection
    results = model(frame)
    
    # Draw grid
    draw_grid(frame)
    
    # Process the results
    for r in results:
        for box in r.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get the class name
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Calculate center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Determine the sector
            sector = get_sector(center_x, center_y, frame.shape)
            
            # Send command to Arduino
            send_command(sector, class_name)
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} - Sector {sector}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Hand Gesture Detection with Sectors', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
ser.close()
```

This final version of the script adds serial communication to send commands to the Arduino. Each command consists of the sector number and the detected gesture.

## Building the Arduino Control System

Now, let's create the Arduino code to control the stepper motors based on the commands received from the Python script.

```cpp
#include <AccelStepper.h>

// Define stepper motors and pins
AccelStepper stepperMotors[6] = {
    AccelStepper(1, 6, 7), 
    AccelStepper(1, 4, 5), 
    AccelStepper(1, 1, 2), 
    AccelStepper(1, 12, 13), 
    AccelStepper(1, 10, 11), 
    AccelStepper(1, 8, 9)
};

const int calibrationStepDelay = 300;   // Microseconds between steps during calibration
const int movementSpeed = 2000;         // Default movement speed
const int movementAcceleration = 800;   // Default movement acceleration
const int lowerPosition = -600;         // open position for the steppers
const int upperPosition = -4000;        // closed for the steppers

const int stepPins[6] = {6, 4, 1, 12, 10, 8};
const int dirPins[6] = {7, 5, 2, 13, 11, 9};
const int limitSwitchPins[6] = {40, 41, 42, 14, 21, 39};

// Calibration function 
void calibrateStepper(int motorIndex) {
    digitalWrite(dirPins[motorIndex], HIGH);

    while (digitalRead(limitSwitchPins[motorIndex]) == 0) {
        digitalWrite(stepPins[motorIndex], HIGH);
        delayMicroseconds(calibrationStepDelay);
        digitalWrite(stepPins[motorIndex], LOW);
        delayMicroseconds(calibrationStepDelay);
    }
    delay(100); // Minor pause

    // Reverse and move away from limit switch
    digitalWrite(dirPins[motorIndex], LOW); // Change direction
    int stepsAway = 50; // Example - Adjust this value as needed
    for (int i = 0; i < stepsAway; i++) {
        digitalWrite(stepPins[motorIndex], HIGH);
        delayMicroseconds(calibrationStepDelay);
        digitalWrite(stepPins[motorIndex], LOW);
        delayMicroseconds(calibrationStepDelay);
    }
    delay(100); // Minor pause
}

bool anyStepperRunning() {
    for (byte i = 0; i < 6; i++) {
        if (stepperMotors[i].distanceToGo() != 0) {
            return true; 
        }
    }
    return false;
}

void setup() {
    Serial.begin(9600);  // Initialize serial communication
    
    for (byte i = 0; i < 6; i++) {
        pinMode(limitSwitchPins[i], INPUT); 
    }

    for (byte i = 0; i < 6; i++) { 
        stepperMotors[i].setMaxSpeed(movementSpeed);
        stepperMotors[i].setAcceleration(movementAcceleration);
        calibrateStepper(i);
        stepperMotors[i].setCurrentPosition(0); 
        stepperMotors[i].moveTo(lowerPosition);
    }

    // Wait for all steppers to reach the open position
    while (anyStepperRunning()) {
        for (byte i = 0; i < 6; i++) {
            stepperMotors[i].run();
        }
    }
}

void loop() {
    // Check for serial commands
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        int commaIndex = command.indexOf(',');
        if (commaIndex != -1) {
            int sector = command.substring(0, commaIndex).toInt();
            String action = command.substring(commaIndex + 1);
            
            if (sector >= 1 && sector <= 6) {
                int motorIndex = sector - 1;
                if (action == "open") {
                    stepperMotors[motorIndex].moveTo(lowerPosition);
                } else if (action == "close") {
                    stepperMotors[motorIndex].moveTo(upperPosition);
                }
            }
        }
    }

    // Non-blocking motor movement
    for (byte i = 0; i < 6; i++) {
        stepperMotors[i].run();
    }
}
```

This Arduino code does the following:

1. Initializes 6 stepper motors and their corresponding pins.
2. Defines a calibration function to set each motor to a known starting position.
3. In the `setup()` function, it calibrates all motors and moves them to the initial open position.
4. In the `loop()` function, it continuously checks for serial commands from the Python script.
5. When a command is received, it moves the corresponding motor to either the open or closed position.
6. The motor movement is non-blocking, allowing multiple motors to move simultaneously.

## Putting It All Together

Now that we have both the Python script for hand gesture detection and the Arduino code for controlling the kinetic sculpture, let's put it all together:

1. Upload the Arduino code to your Arduino board.
2. Connect the stepper motors, drivers, and limit switches to the Arduino as defined in the pin arrays.
3. Run the Python script on your computer.
4. Show hand gestures to the webcam and watch as the kinetic sculpture responds!

## Troubleshooting

If you encounter issues, here are some things to check:

- Ensure all required Python libraries are installed.
- Check that the Arduino is connected to the correct COM port in the Python script.
- Verify that all hardware connections are correct and secure.
- Make sure the custom YOLO model is trained on your specific hand gestures and is located in the correct path.

# **Deploying a PyTorch Model to TensorRT on Jetson Orin Nano**

In this guide, we'll walk through the step-by-step process of converting a PyTorch model developed on a Mac to a TensorRT engine running on the Jetson Orin Nano. We'll also modify the code to ensure it runs smoothly on the Jetson device while interacting with an Arduino. By the end, you'll have replicated your Mac + Arduino setup on the Jetson Orin Nano + Arduino.

---

## **Overview**

- **Objective**: Convert a PyTorch model (`.pt` file) into a TensorRT engine (`.engine` file) and deploy it on the Jetson Orin Nano.
- **JetPack Version**: 5.1.2
- **PyTorch Installation**: Install PyTorch and TorchVision using NVIDIA's Jetson-specific wheel files.
- **Final Outcome**: Run your gesture-controlled kinetic sculpture project on the Jetson Orin Nano, with improved performance using TensorRT acceleration.

---

## **Prerequisites**

- **Hardware**:
  - Jetson Orin Nano Developer Kit
  - Arduino board (e.g., Arduino Uno)
  - USB Camera
  - Stepper motors and necessary hardware for your kinetic sculpture
- **Software**:
  - Original PyTorch model (`yolov8hs.pt`)
  - Arduino code for motor control
- **Knowledge**:
  - Intermediate understanding of PyTorch, TensorRT, and Jetson devices
  - Familiarity with Python and Arduino programming

---

## **Step-by-Step Guide**

### **1. Set Up the Jetson Orin Nano**

#### **1.1. Flash JetPack 5.1.2**

- **Download JetPack 5.1.2** from the [NVIDIA Developer website](https://developer.nvidia.com/embedded/jetpack).
- Use the **NVIDIA SDK Manager** to flash the JetPack onto your Jetson Orin Nano.
- **JetPack 5.1.2** includes:
  - **CUDA 11.x**
  - **cuDNN 8.x**
  - **TensorRT 8.x**

#### **1.2. Initial Configuration**

- Connect the Jetson Orin Nano to a monitor, keyboard, and mouse.
- Boot up the device and complete the on-screen setup instructions.
- Ensure the device is connected to the internet.

---

### **2. Install PyTorch and TorchVision with CUDA Support**

#### **2.1. Remove Any Existing PyTorch Installations**

```bash
sudo pip3 uninstall torch torchvision
```

Run this command multiple times until both packages are fully uninstalled.

#### **2.2. Update System and Install Dependencies**

```bash
sudo apt-get update
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
sudo pip3 install --upgrade pip
sudo pip3 install Cython numpy
```

#### **2.3. Install NVIDIA's PyTorch Build**

Install PyTorch and TorchVision using NVIDIA's Jetson-specific wheels:

```bash
sudo pip3 install torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51.4
```

**Note**: The URL `https://developer.download.nvidia.com/compute/redist/jp/v51.4` corresponds to JetPack 5.1.2.

#### **2.4. Verify PyTorch Installation**

Check if PyTorch is correctly installed and if CUDA is available:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**Expected Output**:

- PyTorch version (e.g., `2.0.0+nv23.05`)
- `True` indicating CUDA is available

---

### **3. Install Other Necessary Python Packages**

```bash
sudo pip3 install ultralytics opencv-python numpy pyserial
```

- **`ultralytics`**: For YOLOv8 support
- **`opencv-python`**: For video processing
- **`pyserial`**: For serial communication with Arduino

---

### **4. Transfer Your PyTorch Model to the Jetson**

Copy your `yolov8hs.pt` model file from your Mac to the Jetson Orin Nano. You can use SCP, a USB drive, or any preferred method.

**Example using SCP**:

On your Mac terminal:

```bash
scp /path/to/yolov8hs.pt orin@<jetson_ip_address>:/home/orin/
```

---

### **5. Export the PyTorch Model to a TensorRT Engine**

#### **5.1. Create an Export Script**

Create a new Python script named `export_model.py` on the Jetson Orin Nano:

```python
from ultralytics import YOLO

# Load your custom YOLOv8 model
model = YOLO('yolov8hs.pt')  # Adjust the path if necessary

# Export the model to TensorRT engine format
model.export(format='engine', device='cuda', half=True)
```

- **`format='engine'`**: Specifies TensorRT engine export
- **`device='cuda'`**: Ensures the model is exported using CUDA
- **`half=True`**: Uses FP16 precision for better performance

#### **5.2. Run the Export Script**

```bash
python3 export_model.py
```

This will generate a `yolov8hs.engine` file in the same directory.

---

### **6. Modify the Python Inference Script**

We'll modify your original inference script to use the TensorRT engine and interact with the Arduino.

#### **6.1. Update the Script**

Create a new script named `inference_trt.py`:

```python
import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time
import torch

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

# Serial communication settings
SERIAL_PORT = '/dev/ttyACM0'  # Adjust based on your Arduino connection
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
    # Load the TensorRT engine
    model = YOLO('yolov8hs.engine')

    # Open serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    frame_count = 0

    # Class names (adjust order if necessary)
    class_names = ['close', 'open']
    colors = {'open': BLUE, 'close': TANGERINE}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        if frame_count % FRAME_PROCESS_INTERVAL == 0:
            # Run inference
            results = model(frame, device='cuda', verbose=False)

            # Process detections
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
```

#### **6.2. Adjust the Class Names**

- Ensure the `class_names` list matches the order of classes in your model.
- If labels are swapped (e.g., "open" when you close your hand), swap the class names:

```python
class_names = ['close', 'open']
```

#### **6.3. Update the Serial Port**

- Replace `'/dev/ttyACM0'` with the correct serial port for your Arduino.
- Use the following command to list available serial ports:

```bash
ls /dev/ttyACM*
```

---

### **7. Connect the Arduino to the Jetson Orin Nano**

- **Hardware Connection**: Connect the Arduino to the Jetson via USB.
- **Arduino Code**: Upload your Arduino motor control code to the Arduino board.
- **Permission Settings**: Ensure the Jetson has permission to access the serial port:

```bash
sudo usermod -a -G dialout $USER
```

- Log out and log back in for the group change to take effect.

---

### **8. Run the Inference Script**

```bash
python3 inference_trt.py
```

- **Webcam Feed**: The script will open a window displaying the webcam feed with detections.
- **Gesture Control**: Use your hand gestures to control the kinetic sculpture.
- **Arduino Interaction**: The script sends commands to the Arduino based on detected gestures.

---

### **9. Troubleshooting and Testing**

#### **9.1. Verify CUDA Availability**

Ensure that CUDA is available in PyTorch:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Should output `True`.

#### **9.2. Camera Access**

If the script cannot access the camera:

- Check if the camera index is correct in `cv2.VideoCapture(0)`. Change the index if necessary.
- Ensure you have permissions to access video devices:

```bash
sudo usermod -aG video $USER
```

Log out and log back in.

#### **9.3. Serial Communication**

- If commands are not reaching the Arduino, check the serial port and baud rate.
- Ensure the Arduino is properly connected and recognized by the Jetson.

---

### **10. Benchmark Performance**

To compare performance between PyTorch and TensorRT:

- **PyTorch Inference**: Run the original script using `yolov8hs.pt`.
- **TensorRT Inference**: Run the modified script using `yolov8hs.engine`.
- **Measure FPS**: Add timing code to measure frames per second (FPS) in both scripts.

**Example Timing Code**:

```python
import time

start_time = time.time()
# Inference code...
end_time = time.time()
fps = 1 / (end_time - start_time)
print(f"FPS: {fps:.2f}")
```

---


## **Additional Resources**

- **NVIDIA Developer Zone**: [Jetson Download Center](https://developer.nvidia.com/embedded/downloads)
- **Ultralytics YOLO**: [Documentation](https://docs.ultralytics.com/)
- **PyTorch for Jetson**: [NVIDIA Forums](https://forums.developer.nvidia.com/c/accelerated-computing/jetson-nano/71)
- **Serial Communication**: [PySerial Documentation](https://pyserial.readthedocs.io/en/latest/)

---

**Note**: Ensure you comply with all software licenses and respect intellectual property when sharing code and models.

---

## Extending the Project

Here are some ideas to extend and improve the project:

1. Add more gesture types for finer control of the sculpture.
2. Implement smooth transitions between gestures.
3. Create a GUI for calibration and control.
4. Add sound effects or LED lighting that responds to gestures.
5. Experiment with different sculpture designs and movements.

## Conclusion

Congratulations! You've now built a real-time hand gesture control system for a kinetic sculpture. This project demonstrates the power of combining computer vision, machine learning, and physical computing to create interactive art installations.

Remember, the key components of this system are:
1. Real-time hand gesture detection using a custom-trained YOLO model.
2. Dividing the camera view into sectors for spatial control.
3. Serial communication between Python and Arduino.
4. Non-blocking control of multiple stepper motors.

This project serves as a foundation that you can build upon and customize to create your own unique interactive experiences. The possibilities are limited only by your imagination!

## Safety Considerations

When working with moving parts and electrical components, always prioritize safety:
- Ensure all wiring is properly insulated and secured.
- Use appropriate power supplies for your motors.
- Be cautious of pinch points in your kinetic sculpture design.
- Always supervise the sculpture when it's in operation.

## Resources for Further Learning

To deepen your understanding of the technologies used in this project, consider exploring these resources:

1. [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
2. [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
3. [Arduino Stepper Library Reference](https://www.arduino.cc/en/Reference/Stepper)

Remember to share your creations and variations on this project. Your work could inspire others in the maker community!

Happy building, and enjoy your new interactive kinetic sculpture!
