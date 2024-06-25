import cv2
import matplotlib.pyplot as plt

def segment_image(path):
    image = cv2.imread(path)
    image_with_boxes = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Extract the character and preprocess it
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi.astype('float32') / 255.0
        roi = roi.reshape((1, 28, 28, 1))
        chars.append((roi, (x, y, w, h)))
    return image_with_boxes, chars