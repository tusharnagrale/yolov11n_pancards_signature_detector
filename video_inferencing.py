import cv2
from ultralytics import YOLO

# Load your trained YOLO model (change path if needed)
model = YOLO("runs/detect/train/weights/best.pt")  # or "yolov8n.pt", "yolov8s.pt", etc.

# Input video path
input_video_path = "signature-s.mp4"  # replace with your video file
output_video_path = "output.mp4"

# Open the video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)  # returns detections

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Write frame to output
    out.write(annotated_frame)

    # Optional: display the video in real-time
    cv2.imshow("YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Output saved to {output_video_path}")
