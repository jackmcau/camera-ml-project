import cv2 as cv
import numpy as np
# I don't use quartz in the main project, pip install pyobjc-framework-Quartz for this file to run
from Quartz import CGEventSourceKeyState, kCGEventSourceStateHIDSystemState

def main():
    frame = 0
    while(True):
        black_image = np.zeros((512,512,3), dtype=np.uint8)
        if frame != 0:
            height, width, _ = black_image.shape
            center_x = 256
            center_y = 256
            cv.circle(black_image, (center_x, center_y), 100, (0, 0, 255), 0)
            cv.circle(black_image, (center_x, center_y), frame, (0, 0, 255), -1)
        cv.imshow("Black Image", black_image)
        if CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, 0x00): # While holding down A, a circle appears
            if frame < 100:
                frame += 1
        else:
            if frame > 0:
                frame -= 1
        if cv.waitKey(1) == ord("q"):
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()