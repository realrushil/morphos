import unittest
import sys
import os
import cv2
import numpy as np

# Add parent directory (face_swap) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_landmarks import process_face_landmarks
from alignment import process_aligned_faces

class TestAlignment(unittest.TestCase):
    def test_process_aligned_faces_no_display(self):
        """Test process_aligned_faces function without display"""
        print("Testing aligned faces detection without display...")
        frame_count = 0
        max_test_frames = 10
        for aligned_face, alignment_data in process_aligned_faces(
            process_face_landmarks(display=False, max_frames=20),
            display=False, max_frames=max_test_frames):
            frame_count += 1
            # Check that we're getting valid data
            self.assertIsNotNone(aligned_face, "Aligned face should not be None")
            self.assertIsInstance(alignment_data, dict, "Alignment data should be a dictionary")
            self.assertIn('frame', alignment_data)
            self.assertIn('landmarks', alignment_data)
            self.assertIn('transform', alignment_data)
            if frame_count >= max_test_frames:
                break
        print(f"No-display test completed. Processed {frame_count} aligned faces.")

def run_display_test():
    """Helper function to run the display test directly"""
    print("Running alignment display test...")
    print("This will show the aligned face window. Press 'q' to quit.")
    for aligned_face, alignment_data in process_aligned_faces(
        process_face_landmarks(display=False),
        display=False, max_frames=None):
        # The display is handled within the generator function
        show_alignment_with_reference(aligned_face)

def show_alignment_with_reference(aligned_face, output_size=(112, 112)):
    ref_pts = np.float32([
        [output_size[0] * 0.3, output_size[1] * 0.4],  # left eye
        [output_size[0] * 0.7, output_size[1] * 0.4],  # right eye
        [output_size[0] * 0.5, output_size[1] * 0.65], # nose tip
    ])
    for pt in ref_pts:
        cv2.circle(aligned_face, (int(pt[0]), int(pt[1])), 2, (0,255,0), -1)
    cv2.imshow('Aligned Face with Reference', aligned_face)
    cv2.waitKey(1)

if __name__ == '__main__':
    print("Alignment Test Options:")
    print("1. Run unittest (automated tests)")
    print("2. Run display test (manual test with video)")
    choice = input("Choose option (1 or 2): ").strip()
    if choice == "1":
        unittest.main()
    elif choice == "2":
        run_display_test()
    else:
        print("Invalid choice. Running display test by default.")
        run_display_test() 