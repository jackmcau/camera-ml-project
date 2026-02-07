from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import mediapipe as mp
import os
import urllib.request

# Download models if they don't exist
def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")

download_model(
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
    'hand_landmarker.task'
)

download_model(
    'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite',
    'blaze_face_short_range.tflite'
)

# Create hand landmarker
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Create face detector
face_options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.5
)
face_detector = vision.FaceDetector.create_from_options(face_options)

def main():

    # Open the camera
    cap = cv.VideoCapture(0)

    if not cap:
        print("Camera could not be opened.")
        exit()

    # Main loop
    try:
        frame_timestamp = 0
        progress_bar = 0
        while(True):
            ret, frame = cap.read()

            if not ret:
                print("Couldn't read frame, exiting")
                break

            
            height, width, _ = frame.shape

            # Convert BGR to RGB
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            frame_timestamp += 33

            # Detect hands
            hand_results = hand_landmarker.detect_for_video(mp_image, frame_timestamp)
            if hand_results.hand_landmarks:
                for hand_landmarks in hand_results.hand_landmarks:
                    # Calculate center and bounding box
                    x_coords = [lm.x * width for lm in hand_landmarks]
                    y_coords = [lm.y * height for lm in hand_landmarks]

                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2

                    # Draw rectangle around hand
                    cv.rectangle(frame, (x_min - 20, y_min - 20), 
                                 (x_max + 20, y_max + 20), (0, 255, 0), 2)

                    # Draw center point
                    cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Detect faces
            face_results = face_detector.detect_for_video(mp_image, frame_timestamp)
            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.bounding_box
                    x = int(bbox.origin_x)
                    y = int(bbox.origin_y)
                    w = int(bbox.width)
                    h = int(bbox.height)

                    #center_x = x + w // 2
                    #center_y = y + h // 2

                    # Draw rectangle around face
                    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Draw center point
                    #cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Overlapping logic
            in_contact = False

            if face_results.detections and hand_results.hand_landmarks:
                face = face_results.detections[0]
                bbox = face.bounding_box
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                w = int(bbox.width)
                h = int(bbox.height)
                for hand_landmarks in hand_results.hand_landmarks:
                    x_coords = [lm.x * width for lm in hand_landmarks]
                    y_coords = [lm.y * height for lm in hand_landmarks]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    # Check the 4 sides for bounding box intersection
                    if (((x_min >= x and x_min <= x+w)  or (x_max >= x and x_max <= x+w)) and
                        ((y_min >= y and y_min <= y+h) or (y_max >= y and y_max <= y+h))):
                        if progress_bar < 50:
                            progress_bar += 1
                            in_contact = True
                            break

            # visualizer and time decay
            if in_contact:
                cv.circle(frame, (width//2, height//2), 50, (0, 0, 255), 0)
                cv.circle(frame, (width//2, height//2), progress_bar, (0, 0, 255), -1)
                if progress_bar == 50:
                    print("50")
            else:
                if progress_bar > 0:
                    progress_bar -= 1

            cv.imshow("Hand and Face Detection", frame)

            if cv.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("Execution interrupted")

    finally:
        print("Cleaning...")
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()