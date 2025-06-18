import cv2
import numpy as np


def process_aligned_faces(face_landmarks_stream, output_size=(112, 112), display=False, max_frames=None):
    """
    Align faces in frames using detected landmarks.

    Args:
        face_landmarks_stream (generator): Generator yielding (frame, face_data) tuples from process_face_landmarks
        output_size (tuple): Desired output size (width, height) for aligned faces
        display (bool): Whether to display the aligned faces in a window
        max_frames (int): Maximum number of aligned faces to process (None for unlimited)

    Yields:
        tuple: (aligned_face, alignment_data) where:
            - aligned_face (numpy.ndarray): The aligned and cropped face image
            - alignment_data (dict): Contains 'frame', 'landmarks', 'transform', and 'bbox'
    """
    # Reference points for alignment (left eye, right eye, nose tip) in output image
    ref_pts = np.float32([
        [output_size[0] * 0.3, output_size[1] * 0.4],  # left eye
        [output_size[0] * 0.7, output_size[1] * 0.4],  # right eye
        [output_size[0] * 0.5, output_size[1] * 0.65], # nose tip
    ])

    frame_count = 0
    try:
        for frame, face_data in face_landmarks_stream:
            landmarks_list = face_data.get('landmarks', [])
            if not landmarks_list:
                continue  # No face detected

            for face_landmarks in landmarks_list:
                # Extract key points: left eye, right eye, nose tip
                # MediaPipe FaceMesh landmark indices:
                # left eye: 33, right eye: 263, nose tip: 1
                pts = np.float32([
                    [face_landmarks.landmark[33].x * frame.shape[1], face_landmarks.landmark[33].y * frame.shape[0]],
                    [face_landmarks.landmark[263].x * frame.shape[1], face_landmarks.landmark[263].y * frame.shape[0]],
                    [face_landmarks.landmark[1].x * frame.shape[1], face_landmarks.landmark[1].y * frame.shape[0]],
                ])
                # Compute affine transform
                M = cv2.getAffineTransform(pts, ref_pts)
                aligned_face = cv2.warpAffine(frame, M, output_size, flags=cv2.INTER_LINEAR)
                alignment_data = {
                    'frame': frame,
                    'landmarks': face_landmarks,
                    'transform': M,
                    'bbox': None  # Optionally, you can compute and store the bounding box
                }
                if display:
                    cv2.imshow('Aligned Face', aligned_face)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        if display:
                            cv2.destroyAllWindows()
                        return
                yield aligned_face, alignment_data
                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    if display:
                        cv2.destroyAllWindows()
                    return
    finally:
        if display:
            cv2.destroyAllWindows() 