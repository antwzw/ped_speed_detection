import cv2
import numpy as np

# Load YOLO model for human detection
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO class names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Capture video
cap = cv2.VideoCapture('/Users/zhengbaoqin/Desktop/shan/speed/v5.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Create a blob from the frame and set it as the input to the network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass
    detections = net.forward(output_layer_names)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'person':
                # Extract bounding box coordinates
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                width = int(obj[2] * frame.shape[1])
                height = int(obj[3] * frame.shape[0])

                x = center_x - width // 2
                y = center_y - height // 2

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(60) == 60:
        break

cv2.destroyAllWindows()
cap.release()
