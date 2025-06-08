import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------- FACE SHAPE ANALYSIS ----------
def calculate_face_shape(landmarks):
    jaw_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17)])
    face_width = distance.euclidean(jaw_points[0], jaw_points[16])
    face_height = distance.euclidean(np.mean(jaw_points[[0, 16]], axis=0),
                                     np.mean([(landmarks.part(19).x, landmarks.part(19).y),
                                              (landmarks.part(24).x, landmarks.part(24).y)], axis=0))
    jaw_width = distance.euclidean(jaw_points[3], jaw_points[13])
    cheekbone_width = distance.euclidean((landmarks.part(1).x, landmarks.part(1).y),
                                         (landmarks.part(15).x, landmarks.part(15).y))
    forehead_width = distance.euclidean((landmarks.part(17).x, landmarks.part(17).y),
                                        (landmarks.part(26).x, landmarks.part(26).y))
    chin_pointiness = distance.euclidean(jaw_points[8], np.mean([jaw_points[7], jaw_points[9]], axis=0))

    width_height_ratio = face_width / face_height if face_height > 0 else 0
    jaw_to_cheekbone_ratio = jaw_width / cheekbone_width if cheekbone_width > 0 else 0
    forehead_to_jaw_ratio = forehead_width / jaw_width if jaw_width > 0 else 0

    if 0.75 < width_height_ratio < 0.85 and jaw_to_cheekbone_ratio < 0.9:
        return "Oval"
    elif 0.8 < width_height_ratio < 0.95 and jaw_to_cheekbone_ratio > 0.9:
        return "Round"
    elif jaw_to_cheekbone_ratio > 0.9 and 0.95 < forehead_to_jaw_ratio < 1.05:
        return "Square"
    elif width_height_ratio < 0.75 and jaw_to_cheekbone_ratio > 0.85:
        return "Rectangle"
    elif forehead_to_jaw_ratio > 1.1 and chin_pointiness > 10:
        return "Heart"
    elif cheekbone_width > face_width and chin_pointiness > 15:
        return "Diamond"
    else:
        return "Undetermined"

# ---------- AGE & GENDER DETECTION ----------
def detect_age_gender(frame, face_cascade, age_net, gender_net, age_labels, gender_labels):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                     (78.4263, 87.7689, 114.8958), swapRB=False, crop=False)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_labels[gender_preds[0].argmax()]

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_labels[age_preds[0].argmax()]

        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame

# ---------- EMOTION DETECTION ----------
def detect_emotion(frame, face_cascade, emotion_model, emotion_labels):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (64, 64))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)  # if model expects (64,64,1)

        prediction = emotion_model.predict(face_img)
        emotion_label = emotion_labels[np.argmax(prediction)]

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    return frame




# ---------- FACE SHAPE DETECTION ----------
def detect_face_shape(frame, dlib_detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = dlib_detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        face_shape = calculate_face_shape(landmarks)

        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        cv2.putText(frame, f"Shape: {face_shape}", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame

# ---------- MAIN ----------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
        gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
    except Exception as e:
        print(f"Error loading age/gender models: {e}")
        cap.release()
        return

    # ✅ Load emotion model without compiling to fix optimizer issues
    try:
        emotion_model = load_model("emotion_model.h5", compile=False)
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        cap.release()
        return

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    age_labels = ['(0-5)', '(6-10)', '(11-15)', '(16-20)', '(21-25)', '(26-30)', '(31-35)', '(36-40)', '(41-100)']
    gender_labels = ['Male', 'Female']

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Error: Could not load Haar cascade.")
        cap.release()
        return

    try:
        dlib_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except Exception as e:
        print(f"Error loading dlib models: {e}")
        cap.release()
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = detect_age_gender(frame, face_cascade, age_net, gender_net, age_labels, gender_labels)
        frame = detect_emotion(frame, face_cascade, emotion_model, emotion_labels)
        frame = detect_face_shape(frame, dlib_detector, predictor)

        cv2.imshow("Live Face Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
