import cv2
import numpy as np
import time

# Load YOLO model for human detection
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO class names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Capture video
cap = cv2.VideoCapture('/Users/zhengbaoqin/Desktop/shan/speed/V6.mp4')

# Initialize variables for speed calculation
person_speeds = []
max_speed = 0  # Maximum speed of persons with sizes larger than the standard size
prev_center_x = None
prev_time = None
standard_person_size = None  # Standard size of the largest person bounding box
min_detection_frames = 5    # Minimum consecutive frames for a valid person detection

# Collect sizes of the first 5 detected persons
collected_sizes = []

first_person_detected = False  # Flag to track the first person detection
time_of_first_appearance = None  # Time of first person appearance

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

    frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get timestamp of the current frame

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

                # Calculate size of the bounding box (assuming size as width * height)
                person_size = width * height

                # Append size to the collected sizes list
                collected_sizes.append(person_size)

                # If this is the first person detection
                if not first_person_detected:
                    time_of_first_appearance = frame_timestamp
                    first_person_detected = True

    cv2.imshow('frame', frame)

    if cv2.waitKey(60) == 60:
        break

# Determine the standard person size as the largest size from the collected sizes
standard_person_size = max(collected_sizes)

# Reset video capture
cap.release()
cap = cv2.VideoCapture('/Users/zhengbaoqin/Desktop/shan/speed/V6.mp4')

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

                # Calculate size of the bounding box (assuming size as width * height)
                person_size = width * height

                # If this is the first or consecutive person detections
                if prev_center_x is not None:
                    # Calculate speed if previous time is available
                    if prev_time is not None:
                        current_time = time.time()
                        time_elapsed = current_time - prev_time
                        speed = abs(center_x - prev_center_x) / time_elapsed

                        # Calculate the average speed for persons with sizes larger than the standard size
                        if person_size > standard_person_size:
                            person_speeds.append(speed)

                        # Update the maximum speed
                        if person_size > standard_person_size and speed > max_speed:
                            max_speed = speed

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                    # Update previous values
                    prev_center_x = center_x
                    prev_time = time.time()
                else:
                    # Initialize previous values
                    prev_center_x = center_x
                    prev_time = time.time()

    cv2.imshow('frame', frame)

    if cv2.waitKey(60) == 60:
        break

cv2.destroyAllWindows()
cap.release()

# Calculate and print the average and max speed of persons with sizes larger than the standard size
if len(person_speeds) >= min_detection_frames:
    average_speed = sum(person_speeds) / len(person_speeds)
    print(f"Average Speed of Detected Persons (Size > Standard Size): {average_speed:.2f} pixels/second")
    print(f"Max Speed of Detected Persons (Size > Standard Size): {max_speed:.2f} pixels/second")

# Print the time of the first person's appearance in the video
if time_of_first_appearance is not None:
    minutes = int(time_of_first_appearance // 60)
    seconds = int(time_of_first_appearance % 60)
    milliseconds = int((time_elapsed % 1) * 1000)
    print(f"Time of First Person Appearance: {minutes} minutes {seconds} seconds {milliseconds} milliseconds")
else:
    print("No valid person detected.")

