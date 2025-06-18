import cv2

def capture_webcam(display=True, max_frames=None):
    """
    Capture live video frames from the webcam.
    
    Args:
        display (bool): Whether to display the webcam feed in a window
        max_frames (int): Maximum number of frames to capture (None for unlimited)
        
    Yields:
        numpy.ndarray: Video frames from the webcam
        
    Raises:
        Exception: If webcam cannot be opened or frame capture fails
    """
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Failed to open webcam")
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to capture frame")
            
            yield frame
            if display:
                cv2.imshow('Webcam', frame)
                
            frame_count += 1
            if (max_frames and frame_count >= max_frames) or (cv2.waitKey(1) & 0xFF == ord('q')):
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()