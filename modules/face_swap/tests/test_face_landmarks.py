import unittest
import sys
import os
import cv2

# Add parent directory (face_swap) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_landmarks import process_face_landmarks

class TestFaceLandmarks(unittest.TestCase):

    def test_process_face_landmarks_display(self):
        """Test process_face_landmarks function with display enabled"""
        print("Testing face landmarks detection with display...")
        print("Press 'q' to quit the test early")
        
        frame_count = 0
        max_test_frames = 50  # Limit test to 50 frames
        
        try:
            for frame, face_data in process_face_landmarks(display=True, max_frames=max_test_frames):
                frame_count += 1
                
                # Check that we're getting valid data
                self.assertIsNotNone(frame, "Frame should not be None")
                self.assertIsInstance(face_data, dict, "Face data should be a dictionary")
                self.assertIn('detections', face_data, "Face data should contain 'detections'")
                self.assertIn('landmarks', face_data, "Face data should contain 'landmarks'")
                
                # Print detection info every 10 frames to reduce output
                if frame_count % 10 == 0:
                    if face_data['detections']:
                        print(f"Frame {frame_count}: {len(face_data['detections'])} face(s) detected")
                    if face_data['landmarks']:
                        print(f"Frame {frame_count}: {len(face_data['landmarks'])} face landmark(s) found")
                
                # Break if we've processed enough frames
                if frame_count >= max_test_frames:
                    break
                    
        except KeyboardInterrupt:
            print(f"\nTest interrupted by user after {frame_count} frames")
        except Exception as e:
            self.fail(f"Test failed with error: {e}")
        finally:
            # Ensure OpenCV windows are properly closed
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Allow time for window cleanup
        
        print(f"\nTest completed. Processed {frame_count} frames.")
        
    def test_process_face_landmarks_no_display(self):
        """Test process_face_landmarks function without display"""
        print("Testing face landmarks detection without display...")
        
        frame_count = 0
        max_test_frames = 10  # Smaller test for no-display mode
        
        for frame, face_data in process_face_landmarks(display=False, max_frames=max_test_frames):
            frame_count += 1
            
            # Verify data structure
            self.assertIsNotNone(frame)
            self.assertIsInstance(face_data, dict)
            self.assertIn('detections', face_data)
            self.assertIn('landmarks', face_data)
            
            if frame_count >= max_test_frames:
                break
        
        print(f"No-display test completed. Processed {frame_count} frames.")

def run_display_test():
    """Helper function to run the display test directly"""
    print("Running face landmarks display test...")
    print("This will show the webcam feed with face detection and landmarks.")
    print("Press 'q' in the video window to quit.")
    
    try:
        for frame, face_data in process_face_landmarks(display=True):
            # The display is handled within the generator function
            pass
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Ensure proper cleanup
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == '__main__':
    # Ask user which test to run
    print("Face Landmarks Test Options:")
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
