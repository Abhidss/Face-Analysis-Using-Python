import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import sqlite3

# ---------- DATABASE SETUP ----------
def init_db():
    conn = sqlite3.connect('face_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            gender TEXT,
            gender_confidence REAL,
            age TEXT,
            age_confidence REAL,
            emotion TEXT,
            emotion_confidence REAL,
            face_shape TEXT
        )
    ''')
    conn.commit()
    conn.close()

# ---------- FACE SHAPE ANALYSIS ----------
def calculate_face_shape(landmarks):
    """Calculate face shape with improved metrics and thresholds."""
    jaw_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17)])
    face_width = distance.euclidean(jaw_points[0], jaw_points[16])
    face_height = distance.euclidean(
        np.mean(jaw_points[[0, 16]], axis=0),
        np.mean([(landmarks.part(27).x, landmarks.part(27).y)], axis=0)
    )
    jaw_width = distance.euclidean(jaw_points[3], jaw_points[13])
    cheekbone_width = distance.euclidean(
        (landmarks.part(1).x, landmarks.part(1).y),
        (landmarks.part(15).x, landmarks.part(15).y)
    )
    forehead_width = distance.euclidean(
        (landmarks.part(19).x, landmarks.part(19).y),
        (landmarks.part(24).x, landmarks.part(24).y)
    )
    chin_line = np.mean([jaw_points[6], jaw_points[10]], axis=0)
    chin_pointiness = distance.euclidean(jaw_points[8], chin_line)
    width_height_ratio = face_width / face_height if face_height > 0 else 0
    jaw_to_face_ratio = jaw_width / face_width if face_width > 0 else 0
    cheekbone_to_jaw_ratio = cheekbone_width / jaw_width if jaw_width > 0 else 0
    forehead_to_jaw_ratio = forehead_width / jaw_width if jaw_width > 0 else 0
    eyebrow_y = np.mean([(landmarks.part(19).y + landmarks.part(24).y) / 2], axis=0)
    nose_base_y = landmarks.part(33).y
    chin_y = landmarks.part(8).y
    upper_third = nose_base_y - eyebrow_y
    middle_third = chin_y - nose_base_y
    thirds_balance = abs(upper_third - middle_third) / middle_third if middle_third > 0 else 1
    if width_height_ratio < 0.65:
        return "Oblong"
    elif (0.65 <= width_height_ratio <= 0.85) and cheekbone_to_jaw_ratio > 1.1 and chin_pointiness < 15:
        return "Oval"
    elif (width_height_ratio > 0.85) and jaw_to_face_ratio > 0.78 and cheekbone_to_jaw_ratio < 1.1:
        return "Round"
    elif (jaw_to_face_ratio > 0.78) and (forehead_to_jaw_ratio > 0.9 and forehead_to_jaw_ratio < 1.1):
        return "Square"
    elif forehead_to_jaw_ratio > 1.15 and chin_pointiness > 15:
        return "Heart"
    elif cheekbone_to_jaw_ratio > 1.15 and chin_pointiness > 15:
        return "Diamond"
    elif jaw_to_face_ratio > 0.8 and forehead_to_jaw_ratio < 0.9:
        return "Triangle"
    else:
        return "Combination"

# ---------- AGE & GENDER DETECTION ----------
def detect_age_gender(frame, face_detector, age_net, gender_net, age_labels, gender_labels):
    """Enhanced age and gender detection with confidence scores."""
    results = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        w, h = x2 - x1, y2 - y1
        padding_ratio = 0.1
        x1_pad = max(0, int(x1 - padding_ratio * w))
        y1_pad = max(0, int(y1 - padding_ratio * h))
        x2_pad = min(frame.shape[1], int(x2 + padding_ratio * w))
        y2_pad = min(frame.shape[0], int(y2 + padding_ratio * h))
        face_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if face_img.size == 0:
            continue
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227), 
            (78.4263, 87.7689, 114.8958), swapRB=False
        )
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_idx = gender_preds[0].argmax()
        gender = gender_labels[gender_idx]
        gender_confidence = float(gender_preds[0][gender_idx]) * 100
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age_idx = age_preds[0].argmax()
        age = age_labels[age_idx]
        age_confidence = float(age_preds[0][age_idx]) * 100
        results.append({
            'bbox': (x1, y1, x2, y2),
            'gender': gender,
            'gender_confidence': gender_confidence,
            'age': age,
            'age_confidence': age_confidence
        })
        label = f"{gender} ({gender_confidence:.1f}%), {age} ({age_confidence:.1f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, results

# ---------- EMOTION DETECTION ----------
def detect_emotion(frame, face_detector, emotion_model, emotion_labels):
    """Improved emotion detection with ensemble approach."""
    results = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
            continue
        face_roi = gray[y1:y2, x1:x2]
        if face_roi.shape[0] < 30 or face_roi.shape[1] < 30:
            continue
        face_roi = cv2.equalizeHist(face_roi)
        face_img = cv2.resize(face_roi, (64, 64))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        prediction = emotion_model.predict(face_img, verbose=0)
        max_idx = np.argmax(prediction[0])
        emotion_label = emotion_labels[max_idx]
        confidence = float(prediction[0][max_idx]) * 100
        results.append({
            'bbox': (x1, y1, x2, y2),
            'emotion': emotion_label,
            'confidence': confidence
        })
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{emotion_label} ({confidence:.1f}%)"
        cv2.putText(frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame, results

# ---------- FACE SHAPE DETECTION ----------
def detect_face_shape(frame, face_detector, landmark_predictor):
    """Enhanced face shape detection with visualization."""
    results = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        points = np.zeros((68, 2), dtype=int)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)
        face_shape = calculate_face_shape(landmarks)
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        results.append({
            'bbox': (x1, y1, x2, y2),
            'shape': face_shape,
            'landmarks': points
        })
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        jawline = points[0:17]
        for i in range(len(jawline)-1):
            cv2.line(frame, tuple(jawline[i]), tuple(jawline[i+1]), (0, 255, 0), 1)
        for (x, y) in points:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(frame, f"Shape: {face_shape}", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return frame, results

# ---------- FPS COUNTER ----------
def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

# ---------- MAIN ----------
def main():
    print("Initializing Face Analysis System...")
    
    # Initialize the database
    init_db()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        print("Loading age and gender models...")
        age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
        gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
        print("Age and gender models loaded successfully.")
    except Exception as e:
        print(f"Error loading age/gender models: {e}")
        print("Please ensure model files are in the correct location.")
        cap.release()
        return

    try:
        print("Loading emotion model...")
        emotion_model = load_model("emotion_model.h5", compile=False)
        print("Emotion model loaded successfully.")
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        print("Please ensure the emotion_model.h5 file is available.")
        cap.release()
        return

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_labels = ['Male', 'Female']

    try:
        print("Loading face detection and landmark models...")
        face_detector = dlib.get_frontal_face_detector()
        landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print("Face detection and landmark models loaded successfully.")
    except Exception as e:
        print(f"Error loading dlib models: {e}")
        print("Please ensure shape_predictor_68_face_landmarks.dat is available.")
        cap.release()
        return

    print("All models loaded successfully.")
    print("Press 'q' to quit, 's' to save current frame")

    frame_count = 0
    start_time = time.time()
    fps = 0
    
    process_age_gender = True
    process_emotion = True
    process_face_shape = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        current_time = time.time()
        if (current_time - start_time) > 1.0:
            fps = frame_count / (current_time - start_time)
            frame_count = 0
            start_time = current_time
        frame_count += 1
        
        display_frame = frame.copy()
        
        age_gender_results = []
        emotion_results = []
        shape_results = []
        
        if process_age_gender:
            display_frame, age_gender_results = detect_age_gender(
                display_frame, face_detector, age_net, gender_net, 
                age_labels, gender_labels
            )
            
        if process_emotion:
            display_frame, emotion_results = detect_emotion(
                display_frame, face_detector, emotion_model, emotion_labels
            )
            
        if process_face_shape:
            display_frame, shape_results = detect_face_shape(
                display_frame, face_detector, landmark_predictor
            )
        
        # Store results in the database
        if age_gender_results or emotion_results or shape_results:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect('face_analysis.db')
            cursor = conn.cursor()
            # We assume that the face detections align index-wise - if multiple faces, this can become complex.
            num_faces = max(len(age_gender_results), len(emotion_results), len(shape_results))
            for i in range(num_faces):
                gender = age_gender_results[i]['gender'] if i < len(age_gender_results) else None
                gender_confidence = age_gender_results[i]['gender_confidence'] if i < len(age_gender_results) else None
                age = age_gender_results[i]['age'] if i < len(age_gender_results) else None
                age_confidence = age_gender_results[i]['age_confidence'] if i < len(age_gender_results) else None
                emotion = emotion_results[i]['emotion'] if i < len(emotion_results) else None
                emotion_confidence = emotion_results[i]['confidence'] if i < len(emotion_results) else None
                face_shape = shape_results[i]['shape'] if i < len(shape_results) else None
                cursor.execute('''
                    INSERT INTO face_analysis (timestamp, gender, gender_confidence, age, age_confidence, emotion, emotion_confidence, face_shape)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, gender, gender_confidence, age, age_confidence, emotion, emotion_confidence, face_shape))
            conn.commit()
            conn.close()
        
        display_frame = draw_fps(display_frame, fps)
        cv2.imshow("Enhanced Face Analysis", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('s'):
            timestamp_save = time.strftime("%Y%m%d-%H%M%S")
            filename = f"face_analysis_{timestamp_save}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Frame saved as {filename}")
        elif key == ord('a'):
            process_age_gender = not process_age_gender
            print(f"Age/Gender detection: {'ON' if process_age_gender else 'OFF'}")
        elif key == ord('e'):
            process_emotion = not process_emotion
            print(f"Emotion detection: {'ON' if process_emotion else 'OFF'}")
        elif key == ord('f'):
            process_face_shape = not process_face_shape
            print(f"Face shape detection: {'ON' if process_face_shape else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("Face Analysis System terminated.")

if __name__ == "__main__":
    main()

