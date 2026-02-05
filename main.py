import tensorflow as tf
import cv2 as cv

cap = cv.VideoCapture(0)

if not cap:
    print("Camera could not be opened.")
    exit()

def main():
    print(tf.__version__)
    imgnum = 0
    try:
        while(True):
            ret, frame = cap.read()

            if not ret:
                print("Couldn't read frame, exiting")
                break
            
            toShow = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            toShow = cv.resize(toShow, (0, 0), fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
            cv.imshow("Camera Feed", toShow)

            if cv.waitKey(1) == ord('q'):
                break
            if cv.waitKey(1) == ord('p'):
                imgnum += 1
                cv.imwrite("screenshots/screenshot{0}.jpg".format(imgnum), toShow)
                print("Screenshot taken")
    except KeyboardInterrupt:
        print("Execution interrupted")
    finally:
        print("Cleaning...")
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()