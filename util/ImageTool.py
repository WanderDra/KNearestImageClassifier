import cv2
import numpy as np


def draw(data):
    img = data.reshape(28, 28, 1).astype(np.uint8)
    print(img)
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
