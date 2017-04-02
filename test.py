import cv2

def main():
    image = cv2.imread('image.jpg')
    gray_image = convert_grayscale(image)
    cv2.imwrite('grayscale.jpg', gray_image)
    equalized_image = apply_histogram_equalization(gray_image)
    cv2.imwrite('histogram_eq.jpg', equalized_image)

def convert_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

if __name__ == "__main__":
    main()
