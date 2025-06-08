import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import sys

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

    # Debugging output
    print(f"Measurements: Face Width: {face_width:.2f}, Face Height: {face_height:.2f}, W/H Ratio: {width_height_ratio:.2f}")
    print(f"Jaw Width: {jaw_width:.2f}, Cheekbone Width: {cheekbone_width:.2f}, J/C Ratio: {jaw_to_cheekbone_ratio:.2f}")
    print(f"Forehead Width: {forehead_width:.2f}, F/J Ratio: {forehead_to_jaw_ratio:.2f}, Chin Pointiness: {chin_pointiness:.2f}")

    # Face shape classification with normalized confidence
    shapes = ["Oval", "Round", "Square", "Rectangle", "Heart", "Diamond", "Triangle"]
    scores = []

    # Oval: Balanced proportions, longer than wide
    if 0.65 <= width_height_ratio <= 0.90 and jaw_to_cheekbone_ratio < 0.95:
        score = 1.0 - (abs(width_height_ratio - 0.80) / 0.25 + abs(jaw_to_cheekbone_ratio - 0.85) / 0.2)
        scores.append(("Oval", max(0.0, score)))
    else:
        scores.append(("Oval", 0.0))

    # Round: Similar width and height
    if 0.85 <= width_height_ratio <= 1.10 and jaw_to_cheekbone_ratio >= 0.95:
        score = 1.0 - (abs(width_height_ratio - 1.0) / 0.25 + abs(jaw_to_cheekbone_ratio - 1.0) / 0.2)
        scores.append(("Round", max(0.0, score)))
    else:
        scores.append(("Round", 0.0))

    # Square: Equal widths, tighter range
    if 0.95 <= jaw_to_cheekbone_ratio <= 1.05 and 0.95 <= forehead_to_jaw_ratio <= 1.05:
        score = 1.0 - (abs(jaw_to_cheekbone_ratio - 1.0) / 0.1 + abs(forehead_to_jaw_ratio - 1.0) / 0.1)
        scores.append(("Square", max(0.0, score)))
    else:
        scores.append(("Square", 0.0))

    # Rectangle: Longer than wide
    if width_height_ratio < 0.70 and jaw_to_cheekbone_ratio >= 0.85:
        score = 1.0 - (abs(width_height_ratio - 0.60) / 0.2 + abs(jaw_to_cheekbone_ratio - 0.90) / 0.2)
        scores.append(("Rectangle", max(0.0, score)))
    else:
        scores.append(("Rectangle", 0.0))

    # Heart: Wide forehead, pointed chin
    if forehead_to_jaw_ratio > 1.10 and chin_pointiness > 8:
        score = 1.0 - (abs(forehead_to_jaw_ratio - 1.20) / 0.2 + abs(chin_pointiness - 10) / 5.0)
        scores.append(("Heart", max(0.0, score)))
    else:
        scores.append(("Heart", 0.0))

    # Diamond: Wide cheekbones, pointed chin
    if cheekbone_width > face_width * 1.05 and chin_pointiness > 10:
        score = 1.0 - (abs(cheekbone_width / face_width - 1.10) / 0.2 + abs(chin_pointiness - 12) / 5.0)
        scores.append(("Diamond", max(0.0, score)))
    else:
        scores.append(("Diamond", 0.0))

    # Triangle: Narrow forehead, wide jaw
    if width_height_ratio < 0.70 and chin_pointiness > 12:
        score = 1.0 - (abs(width_height_ratio - 0.60) / 0.2 + abs(chin_pointiness - 15) / 5.0)
        scores.append(("Triangle", max(0.0, score)))
    else:
        scores.append(("Triangle", 0.0))

    # Normalize scores
    total_score = sum(score for _, score in scores)
    if total_score > 0:
        scores = [(shape, score / total_score) for shape, score in scores]

    # Find best shape
    best_shape, best_score = max(scores, key=lambda x: x[1])
    confidence = best_score * 100

    # Debugging: Print all scores
    print("Shape Scores:")
    for shape, score in scores:
        print(f"  {shape}: {score*100:.1f}%")
    print(f"Selected Shape: {best_shape}, Confidence: {confidence:.1f}%")

    if best_score < 0.2:  # Low confidence threshold
        print("Undetermined: Confidence too low. Likely causes: Non-frontal face, poor lighting, or inaccurate landmarks.")
        return "Undetermined", confidence

    return best_shape, confidence

# ---------- FACE ALIGNMENT ----------
def align_face(image, landmarks):
    left_eye = np.mean([(landmarks.part(36).x, landmarks.part(36).y),
                        (landmarks.part(39).x, landmarks.part(39).y)], axis=0)
    right_eye = np.mean([(landmarks.part(42).x, landmarks.part(42).y),
                         (landmarks.part(45).x, landmarks.part(45).y)], axis=0)
    
    dX = right_eye[0] - left_eye[0]
    dY = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dY, dX))
    
    desired_right_eye_x = 0.35
    desired_left_eye_x = 1.0 - desired_right_eye_x
    desired_face_width = 227
    desired_face_height = 227
    
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    return aligned

# ---------- AGE & GENDER DETECTION ----------
def detect_age_gender(frame, face_cascade, age_net, gender_net, age_labels, gender_labels, dlib_detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        dlib_rect = dlib.rectangle(x, y, x+w, y+h)
        landmarks = predictor(gray, dlib_rect)
        aligned_face = align_face(face_img, landmarks)
        
        blob = cv2.dnn.blobFromImage(aligned_face, 1.0, (227, 227),
                                     (78.4263, 87.7689, 114.8958), swapRB=False)
        
        gender_net.setInput(blob)
        gender_probs = gender_net.forward()[0]
        gender_idx = gender_probs.argmax()
        gender = gender_labels[gender_idx]
        gender_conf = gender_probs[gender_idx] * 100
        
        age_net.setInput(blob)
        age_probs = age_net.forward()[0]
        age_idx = age_probs.argmax()
        age = age_labels[age_idx]
        age_conf = age_probs[age_idx] * 100
        
        label = f"{gender} ({gender_conf:.1f}%), {age} ({age_conf:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

# ---------- FACE SHAPE DETECTION ----------
def detect_face_shape(frame, dlib_detector, predictor, prev_landmarks=None, alpha=0.4):
    # Preprocess frame for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Normalize brightness
    gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Apply adaptive thresholding
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    faces = dlib_detector(gray, 1)  # Upsample for better detection
    
    for face in faces:
        landmarks = predictor(gray, face)
        if prev_landmarks is not None:
            # Smoother landmark transition
            for i in range(68):
                x = int(alpha * landmarks.part(i).x + (1 - alpha) * prev_landmarks.part(i).x)
                y = int(alpha * landmarks.part(i).y + (1 - alpha) * prev_landmarks.part(i).y)
                landmarks.part(i).x = x
                landmarks.part(i).y = y
        
        face_shape, confidence = calculate_face_shape(landmarks)
        
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        label = f"Shape: {face_shape} ({confidence:.1f}%)"
        cv2.putText(frame, label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame, landmarks
    
    return frame, prev_landmarks

# ---------- MAIN ----------
def main():
    try:
        age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
        gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
    except Exception as e:
        print(f"Error loading DNN models: {e}")
        sys.exit(1)
    
    age_labels = ['(0-5)', '(6-10)', '(11-15)', '(16-20)', '(21-25)', '(26-30)', '(31-35)', '(36-40)', '(41-100)']
    gender_labels = ['Male', 'Female']
    
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if face_cascade.empty():
            raise Exception("Failed to load Haar cascade")
    except Exception as e:
        print(f"Error loading Haar cascade: {e}")
        sys.exit(1)
    
    try:
        dlib_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except Exception as e:
        print(f"Error loading dlib models: {e}")
        sys.exit(1)
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Cannot open camera")
    except Exception as e:
        print(f"Error opening camera: {e}")
        sys.exit(1)
    
    print("Press 'q' to quit")
    prev_landmarks = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = detect_age_gender(frame, face_cascade, age_net, gender_net, age_labels, gender_labels, dlib_detector, predictor)
        frame, prev_landmarks = detect_face_shape(frame, dlib_detector, predictor, prev_landmarks)
        
        cv2.imshow("Live Face Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()