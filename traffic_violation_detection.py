# import necessary libraries
import cv2
import numpy as np
import easyocr
from pytesseract import image_to_string
# from google.colab.patches import cv2_imshow




# Function to preprocess the frame
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised_frame = cv2.bilateralFilter(gray_frame, 9, 75, 75)
    return denoised_frame


# Function to detect the color of the traffic light
def detect_traffic_light_color(frame, light_rect):

    # Determining the location of the turn signal in the image
    light_roi = frame[light_rect[1]:light_rect[3], light_rect[0]:light_rect[2]]

    # Convert to HSV color space
    hsv_roi = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)

    # Determine the range of red color
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv_roi, lower_red, upper_red)

    # Checking for the presence of red color
    if np.any(red_mask):
        return 0 # Red light

    # You can add similar conditions for other colors as well
    return -1 # Uncertain status

# Function to create a triangular mask for the region of interest
def create_mask(roi):
    height, width = roi.shape[:2]
    mask = np.zeros_like(roi)
    triangle = np.array([[(0, height), (width / 2, height / 2), (width, height)]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    return mask


# Function to detect the white boundary line
def detect_white_line(frame, roi_rect):
    roi = frame[roi_rect[1]:roi_rect[3], roi_rect[0]:roi_rect[2]]
    
    # Define a rectangular mask for the desired area
    mask = np.zeros_like(roi)
    points = np.array([[50, 0], [roi.shape[1] - 50, 0], [roi.shape[1], roi.shape[0]], [0, roi.shape[0]]])
    cv2.fillPoly(mask, [points], 255)
    
    # Apply mask on ROI
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask[:,:,0])
    
    # Image processing and line detection as before
    gray_roi = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges_roi = cv2.Canny(blurred_roi, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_roi = cv2.dilate(edges_roi, kernel, iterations=2)
    lines = cv2.HoughLinesP(dilated_roi, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=30)

    # Filtration of lines based on slope
    if lines is not None:
        filtered_lines = [line for line in lines if abs((line[0][3] - line[0][1]) / (line[0][2] - line[0][0])) < 0.1]
        avg_line = np.mean(filtered_lines, axis=0).astype(int)
        avg_line += (roi_rect[0], roi_rect[1], roi_rect[0], roi_rect[1])
        return avg_line[0]
    return None




# Function to find the equation of the line
def find_line_equation(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


# Function to create a mask from a line
def create_mask_from_line(frame_shape, m, b):
    mask = np.ones(frame_shape[:2], dtype=np.uint8) * 255
    height, width = frame_shape[:2]
    for x in range(width):
        y_line = int(m * x + b)
        mask[:y_line, x] = 0 # Blacken the part that is less than the line
    return mask


# Function to detect license plates using a pre-trained Haar cascade
def detect_license_plate_using_cascade(region_of_interest):
    license_plate_cascade_path = 'haarcascade_license_plate_rus_16stages.xml'
    license_plate_cascade = cv2.CascadeClassifier(license_plate_cascade_path)

    plates = license_plate_cascade.detectMultiScale(region_of_interest, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [region_of_interest[y:y+h, x:x+w] for x, y, w, h in plates]



# Function to recognize license plate using OCR
def recognize_license_plate(plate_image):
    plate_number = image_to_string(plate_image, config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return plate_number



# def preprocess_license_plate(plate):
#     gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 100, 200)
#     return edges


def enhance_plate_contrast(plate_image):
    # Increase image contrast using CLAHE transform
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_plate = clahe.apply(plate_image)
    return enhanced_plate


def recognize_license_plate(plate_image):
    # Turn to gray
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    # Contrast enhancement using CLAHE transform
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_plate = clahe.apply(gray)
    # Noise removal with averaging filter
    denoised_plate = cv2.medianBlur(enhanced_plate, 3)
    # Increase the sharpness of the edges with the Sharpen filter
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_plate = cv2.filter2D(denoised_plate, -1, sharpening_kernel)
    
    # Use easyocr to recognize the text
    reader = easyocr.Reader(lang_list=['en'], gpu=False)
    result = reader.readtext(sharpened_plate)
    plate_number = " ".join([item[1] for item in result])
    return plate_number

def register_violation_and_fine(license_plate_number):
    fine_amount = 200
    print(f"License plate {license_plate_number} has been fined ${fine_amount} for violating red light!")
    return fine_amount



def main():
    video_path = 'traffic_video_modified.mp4'
    light_rect = (1750, 0, 1920, 350) # Adjustment based on the actual position of the light
    line_roi_rect = (600, 820, 1920, 915) # Location of the boundary line
    cap = cv2.VideoCapture(video_path)
    fined_plates = set()
    if not cap.isOpened():
        print("Error: Unable to open video.")
        exit()

    violations_file = open('violations.txt', 'a')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame)
        traffic_light_status = detect_traffic_light_color(frame, light_rect)
        boundary_line = detect_white_line(frame, line_roi_rect)

        # Draw a boundary line and create a mask
        if boundary_line is not None:
            x1, y1, x2, y2 = boundary_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            m, b = find_line_equation(x1, y1, x2, y2)
            mask = create_mask_from_line(frame.shape, m, b)
            region_of_interest = cv2.bitwise_and(frame, frame, mask=mask)

            # If the light is red, we detect license plates
            if traffic_light_status == 0:
                detected_plates = detect_license_plate_using_cascade(region_of_interest)[:5]
                for plate_image in detected_plates:
                    license_plate_number = recognize_license_plate(plate_image)
                    if license_plate_number not in fined_plates: # If the license plate has not already been fined
                        fined_plates.add(license_plate_number) # Adding plates to the collection of fined ones
                        print(f"Car with license plate {license_plate_number} has violated red light!")
                        fine_amount = register_violation_and_fine(license_plate_number)
                        violations_file.write(f"License Plate: {license_plate_number}, Fined: ${fine_amount}\n")
                        cv2.imshow('plate_image',plate_image)
            cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    violations_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

