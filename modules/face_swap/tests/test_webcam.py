import unittest
from unittest.mock import MagicMock, patch
import cv2
import sys
import os

# Add parent directory (face_swap) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from webcam import capture_webcam  # Direct import now works

class TestWebcamCapture(unittest.TestCase):

    @patch('cv2.VideoCapture')
    def test_capture_webcam(self, mock_video_capture):
        # Mock the VideoCapture object
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap

        # Set up the mock to return True for isOpened and read
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, 'frame')

        # Call the function
        capture_webcam(display=False, max_frames=5)

        # Assertions to ensure the methods were called
        mock_cap.isOpened.assert_called_once()
        mock_cap.read.assert_called()
        mock_cap.release.assert_called_once()

if __name__ == '__main__':
    unittest.main()