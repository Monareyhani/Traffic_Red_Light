import cv2
import numpy as np

def preprocess_frame(frame):
    # تبدیل به تصویر خاکستری
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # حذف نویز با فیلتر بی‌طرفی
    denoised_frame = cv2.bilateralFilter(gray_frame, 9, 75, 75)
    return denoised_frame

def detect_traffic_light_color(frame, light_rect):
    # تعیین موقعیت چراغ راهنما در تصویر
    light_roi = frame[light_rect[1]:light_rect[3], light_rect[0]:light_rect[2]]
    # تبدیل به فضای رنگ HSV
    hsv_roi = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
    # تعیین محدوده رنگ قرمز
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv_roi, lower_red, upper_red)
    # بررسی وجود رنگ قرمز
    if np.any(red_mask):
        return 0 # چراغ قرمز
    # می‌توانید برای سایر رنگ‌ها نیز شرایط مشابهی اضافه کنید
    return -1 # وضعیت نامشخص


def create_mask(roi):
    height, width = roi.shape[:2]
    mask = np.zeros_like(roi)
    triangle = np.array([[(0, height), (width / 2, height / 2), (width, height)]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    return mask


def detect_white_line(frame, roi_rect):
    roi = frame[roi_rect[1]:roi_rect[3], roi_rect[0]:roi_rect[2]]
    
    # تعریف یک ماسک با شکل چهارضلعی برای منطقه مورد نظر
    mask = np.zeros_like(roi)
    points = np.array([[50, 0], [roi.shape[1] - 50, 0], [roi.shape[1], roi.shape[0]], [0, roi.shape[0]]])
    cv2.fillPoly(mask, [points], 255)
    
    # اعمال ماسک بر روی ROI
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask[:,:,0])
    
    # پردازش تصویر و تشخیص خطوط مانند قبل
    gray_roi = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges_roi = cv2.Canny(blurred_roi, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_roi = cv2.dilate(edges_roi, kernel, iterations=2)
    lines = cv2.HoughLinesP(dilated_roi, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=30)

    # فیلتراسیون خطوط بر اساس شیب
    if lines is not None:
        filtered_lines = [line for line in lines if abs((line[0][3] - line[0][1]) / (line[0][2] - line[0][0])) < 0.1]
        avg_line = np.mean(filtered_lines, axis=0).astype(int)
        avg_line += (roi_rect[0], roi_rect[1], roi_rect[0], roi_rect[1])
        return avg_line[0]
    return None







def find_line_equation(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b


def create_mask_from_line(frame_shape, m, b):
    mask = np.ones(frame_shape[:2], dtype=np.uint8) * 255
    height, width = frame_shape[:2]
    for x in range(width):
        y_line = int(m * x + b)
        mask[:y_line, x] = 0 # مشکی کردن قسمتی که کمتر از خط است
    return mask


def detect_license_plate_using_cascade(region_of_interest):
    license_plate_cascade_path = 'haarcascade_license_plate_rus_16stages.xml'
    license_plate_cascade = cv2.CascadeClassifier(license_plate_cascade_path)

    plates = license_plate_cascade.detectMultiScale(region_of_interest, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [region_of_interest[y:y+h, x:x+w] for x, y, w, h in plates]

from pytesseract import image_to_string

def recognize_license_plate(plate_image):
    plate_number = image_to_string(plate_image, config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return plate_number

def preprocess_license_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges


def enhance_plate_contrast(plate_image):
    # افزایش کنتراست تصویر با استفاده از تبدیل CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_plate = clahe.apply(plate_image)
    return enhanced_plate

from pytesseract import image_to_string

def recognize_license_plate(plate_image):
    # پیش‌پردازش تصویر پلاک
    preprocessed_plate = preprocess_license_plate(plate_image)
    enhanced_plate = enhance_plate_contrast(preprocessed_plate)
    
    # تست تنظیمات Tesseract
    plate_number = image_to_string(enhanced_plate, config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return plate_number

def register_violation_and_fine(license_plate_number):
    fine_amount = 200
    print(f"License plate {license_plate_number} has been fined ${fine_amount} for violating red light!")
    return fine_amount



def main():
    video_path = 'traffic_video_modified.mp4'
    light_rect = (1750, 0, 1920, 350) # تنظیم براساس موقعیت واقعی چراغ
    line_roi_rect = (600, 820, 1920, 915) # موقعیت خط مرزی
    cap = cv2.VideoCapture(video_path)
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

        # رسم خط مرزی و ایجاد ماسک
        if boundary_line is not None:
            x1, y1, x2, y2 = boundary_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            m, b = find_line_equation(x1, y1, x2, y2)
            mask = create_mask_from_line(frame.shape, m, b)
            region_of_interest = cv2.bitwise_and(frame, frame, mask=mask)

            # اگر چراغ قرمز است، پلاک‌ها را تشخیص می‌دهیم
            if traffic_light_status == 0:
                detected_plates = detect_license_plate_using_cascade(region_of_interest)
                for plate_image in detected_plates:
                    license_plate_number = recognize_license_plate(plate_image)
                    print(f"Car with license plate {license_plate_number} has violated red light!")
                    fine_amount = register_violation_and_fine(license_plate_number)
                    violations_file.write(f"License Plate: {license_plate_number}, Fined: ${fine_amount}\n")
                    cv2.imshow('License Plate', plate_image)

        cv2.imshow('Result Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    violations_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()