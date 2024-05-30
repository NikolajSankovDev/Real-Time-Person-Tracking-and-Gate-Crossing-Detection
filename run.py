import cv2
import torch
import datetime
from torchvision import transforms, models
import torch.nn as nn
from ultralytics import YOLO
import time

# Points for cropping
point1 = (226, 172)  # Top-left corner
point2 = (980, 929)  # Bottom-right corner

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2*2) 
model.load_state_dict(torch.load('keypoints_model_cropped.pth', map_location=device))
model = model.to(device)
model.eval()

# Load the YOLO model
yolo_model = YOLO('yolov8x')

# Setup video stream, you can use video stream from IP-camera or run it with a pre-recorded video file
# cap = cv2.VideoCapture('rtsp://your_rtsp_stream_here')
cap = cv2.VideoCapture('input_videos/video.avi')

# Get the FPS (Frames Per Second) of the stream
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"stream is {fps} fps")

def get_gate_keypoints(frame, model, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    
    kps = outputs[0].cpu().numpy().flatten()

    h, w = frame.shape[:2]
    kps[::2] *= w / 640.0  # Rescale x coordinates
    kps[1::2] *= h / 640.0 # Rescale y coordinates

    return kps.reshape(-1, 2)

# Main processing loop
ret, first_frame = cap.read()
first_frame = first_frame[point1[1]:point2[1], point1[0]:point2[0]]

if not ret:
    print("Failed to grab the first frame.")
    cap.release()
    exit()
else:
    print('Got frame')
gate_points = get_gate_keypoints(first_frame, model, device)
x1, y1 = gate_points[0]  # Starting point of the gate
x2, y2 = gate_points[1]  # Ending point of the gate
print(x1, y1)
print(x2, y2)

person_status = {}
last_crossing_time = {}
debounce_duration = 0.1  # Time in seconds

def process_frame(frame):
    start_time = time.time()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model.track(frame_rgb, persist=True, verbose=False, tracker="custom_tracker.yaml")

    for det in results[0].boxes:
        if det.cls[0] == 0:  # class '0' is for persons
            x, y, w, h = det.xywh[0]
            x, y, x3, y3 = det.xyxy[0]
            confidence = det.conf[0]
            if det.id == None:
                continue
            person_id = int(det.id[0].item())
            xf = int((x3-x) / 2 + x)
            yf = int(y3 - (y3-y)*0.05)

            # Draw bounding box and a dot at (xf, yf)
            cv2.rectangle(frame, (int(x), int(y)), (int(x3), int(y3)), (0, 255, 0), 2)
            cv2.circle(frame, (int(xf), int(yf)), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.putText(frame, f"ID: {person_id} {confidence:.2f}", (int(x), int(y) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            current_time = time.time()
            if person_id in last_crossing_time and (current_time - last_crossing_time[person_id]) < debounce_duration:
                continue  # Skip the crossing logic if within debounce duration
            # Gate line crossing logic
            if person_id not in person_status:
                if x1 <= xf <= x2:
                    current_status = 'inside' if yf > y1 else 'outside'
                    person_status[person_id] = current_status
                    print('set status to', current_status)
            else:
                current_status = 'inside' if yf > y1 else 'outside'
                if person_status[person_id] != current_status:
                    print(f"Event: {person_id} crossed the gate at {datetime.datetime.now()} from {person_status[person_id]} to {current_status}")
                    person_status[person_id] = current_status
                    print('set status to', person_status[person_id])
                    last_crossing_time[person_id] = current_time

    cv2.imshow('Frame', frame)
    cv2.waitKey(1)  # Display a frame for 1 ms, then go to the next frame

    elapsed_time = time.time() - start_time
    return elapsed_time

frame_count = 0
total_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[point1[1]:point2[1], point1[0]:point2[0]]
        elapsed_time = process_frame(frame)
        # Update FPS calculation
        frame_count += 1
        total_time += elapsed_time
        # Calculate and display the average FPS every 100 frames
        if frame_count % 100 == 0:
            average_fps = frame_count / total_time

finally:
    cap.release()
    cv2.destroyAllWindows()
