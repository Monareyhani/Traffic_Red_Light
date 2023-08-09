# Traffic Violation Detection System

## Introduction

This code is part of a traffic violation detection system that processes a video of a road intersection to detect vehicles that cross the stop line when the traffic light is red. It uses computer vision techniques to detect the traffic light color, locate the boundary line, and recognize license plates of violating vehicles.

## Features

- **Traffic Light Color Detection**: Determines the current color of the traffic light.
- **Boundary Line Detection**: Locates the boundary line that vehicles must not cross during a red light.
- **License Plate Recognition**: Recognizes the license plates of vehicles that violate the red light.
- **Fine Registration**: Records the violations and imposes fines.

## Dependencies

- OpenCV (cv2): For image processing and video handling.
- NumPy: For numerical operations.
- easyOCR: For optical character recognition (OCR) on license plates.
- pytesseract (optional): An alternative OCR tool.

## How to Run

1. **Clone the Repository**: Clone this repository to your local machine.

2. **Install Dependencies**: Install the required dependencies using the following commands:

   ```bash
   pip install opencv-python-headless numpy easyocr pytesseract
   ```

3. **Set Paths**: Update the paths to the video file and Haar cascade for license plate detection in the `main()` function.

4. **Run the Code**: Execute the script using the following command:

   ```bash
   python traffic_violation_detection.py
   ```

5. **View Results**: The code will process the video and print the license plates of vehicles that have violated the red light. The violations will also be recorded in a text file named `violations.txt`.

## Notes

- Ensure that the coordinates for the traffic light and the boundary line region of interest (ROI) are correctly set in the `main()` function based on the actual positions in your video.
- The Haar cascade file for license plate detection must be placed in the specified path or updated accordingly.

## License

This project is open-source and available under the [MIT License](LICENSE).
