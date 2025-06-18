import cv2
import mediapipe as mp
from webcam import capture_webcam

def process_face_landmarks(display=False, max_frames=None):
    """
    Process webcam frames and detect face landmarks.
    
    Args:
        display (bool): Whether to display the processed frames
        max_frames (int): Maximum number of frames to process (None for unlimited)
        
    Yields:
        tuple: (frame, face_data) where face_data contains:
            - 'detections': face detection results
            - 'landmarks': face mesh landmarks
    """
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Optimized settings for better performance
    face_detection = mp_face_detection.FaceDetection(
        min_detection_confidence=0.1,  # Lower confidence for speed
        model_selection=0  # Use model 0 (faster, shorter range)
    )
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Only detect one face for speed
        refine_landmarks=False,  # Disable refinement for speed
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1
    )

    try:
        for frame in capture_webcam(display=False, max_frames=max_frames):
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results_detection = face_detection.process(image_rgb)
            results_mesh = face_mesh.process(image_rgb)

            # Prepare face data
            face_data = {
                'detections': results_detection.detections if results_detection.detections else [],
                'landmarks': results_mesh.multi_face_landmarks if results_mesh.multi_face_landmarks else []
            }

            # Draw visualizations if display is enabled
            if display:
                if results_detection.detections:
                    for detection in results_detection.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(frame, bbox, (0, 255, 0), 2)

                if results_mesh.multi_face_landmarks:
                    for face_landmarks in results_mesh.multi_face_landmarks:
                        # Draw fewer landmarks for better performance
                        mp_drawing.draw_landmarks(
                            frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=0),
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )

                cv2.imshow('MediaPipe Face Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Yield frame and face data
            yield frame, face_data

    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if display:
            cv2.destroyAllWindows()