import cv2
import numpy as np


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # gray scale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    canny_img = cv2.Canny(blur, 50, 150)  # detect sharp changes in color intensity (edge detection)
    # cv2.Canny(image, low_threshold, high_threshold) [1:3 ratio]
    return canny_img


def display_lines(img, _lines):
    _line_image = np.zeros_like(img)  # Make black image to display lines on

    if _lines is not None:  # Make sure that there are lines detected
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # Find the coordinated of each line
            cv2.line(_line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # Draw on the black image with blue color (BGR)
    return _line_image


def region_of_interest(img):
    height = image.shape[0]  # Get the height of the image
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]  # vertices of triangle (lane)
    ])
    mask = np.zeros_like(img)  # Create a black mask
    cv2.fillPoly(mask, polygons, 255)  # Put the triangle on top of the mask

    # Mask the canny image to show the region of interest traced by the trialing mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


image = cv2.imread("test_image.jpg")  # Load the image
lane_image = np.copy(image)
canny = canny(image)
cropped_image = region_of_interest(canny)
# Find straight lines
lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), 40, 5)
lines_image = display_lines(lane_image, lines)
final_image = cv2.addWeighted(lane_image, 0.8, lines_image, 1, 1)  # Overlay lines on the original image

cv2.imshow("results", final_image)  # Show the image
cv2.waitKey(0)
