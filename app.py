import cv2
from ultralytics import YOLO

# Load a larger YOLOv8 model from the ultralytics hub for better accuracy
model = YOLO('yolov8x')  # Try 'yolov8x' for higher accuracy, 'yolov8l' if too slow

# Open the video file
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output_video.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # Restart the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Perform inference with YOLOv8
    results = model(frame)

    # Parse and draw detections
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            # Draw bounding box and label on the frame
            label = f'Car: {confidence:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
