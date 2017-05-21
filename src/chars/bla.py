import cv2

img = cv2.imread("1.png")
height = img.shape[0]
width = img.shape[1]

if width > height:
    border_width = 0
    border_height = (width - height) / 2
else:
    border_height = 0
    border_width = (height - width) / 2

border = cv2.copyMakeBorder(img, top=border_height, bottom=border_height, left=border_width, right=border_width, borderType=cv2.BORDER_CONSTANT, value=[ 255, 255, 255])

resized = cv2.resize(border, (300, 300))

cv2.imwrite("bla.png", resized)
