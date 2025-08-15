import cv2
from ultralytics import YOLO

def traffic_analysis_yolo_bytetrack(video_path, output_path="output_traffic.mp4"):
    model = YOLO("yolo11n.pt")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can use 'XVID' or 'MJPG'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Starting vehicle detection and tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
            
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, iou=0.5, show=False)

        # Process results
        if results and results[0].boxes.id is not None:
            # Get annotated frame with bounding boxes and track IDs
            annotated_frame = results[0].plot()

            # Display the frame
            cv2.imshow("Traffic Analysis", annotated_frame)
            
            out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Traffic analysis completed. Output saved to {output_path}")

if __name__ == "__main__":
    # Replace with your video file path
    input_video = "road_traffic.mp4" 
    
    traffic_analysis_yolo_bytetrack(input_video)