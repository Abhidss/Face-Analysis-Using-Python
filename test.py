import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Paths to model files
age_model = 'age_net.caffemodel'
age_proto = 'age_deploy.prototxt'
gender_model = 'gender_net.caffemodel'
gender_proto = 'gender_deploy.prototxt'

# Load pre-trained models
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Age and gender labels
age_labels = ['(0-5)', '(6-10)', '(11-15)', '(16-20)', '(21-25)','(26-30)', '(31-35)', '(36-40)', '(41-100)']
gender_labels = ['Male', 'Female']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = frame[y:y + h, x:x + w]

        # Prepare input for deep learning models
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_pred = gender_net.forward()
        gender = gender_labels[gender_pred[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_pred = age_net.forward()
        age = age_labels[age_pred[0].argmax()]

        # Draw bounding box and label
        label = f'{gender}, {age}'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        def calculate_face_shape(landmarks):
              """
    Calculate face shape based on facial landmarks
    Returns one of: 'Oval', 'Round', 'Square', 'Rectangle', 'Heart', 'Diamond'
    """
    
     # Extract key points
    jaw_points = []
    for i in range(0, 17):  # Jaw line landmarks
        jaw_points.append((landmarks.part(i).x, landmarks.part(i).y))
    
    jaw_points = np.array(jaw_points)
    
    # Face measurements
    face_width = distance.euclidean(jaw_points[0], jaw_points[16])  # Width at jawline
    face_height = distance.euclidean(
        np.mean(jaw_points[[0, 16]], axis=0),  # Midpoint of jawline
        np.mean([(landmarks.part(19).x, landmarks.part(19).y), 
                (landmarks.part(24).x, landmarks.part(24).y)], axis=0)  # Midpoint of eyebrows
    )
    
    
      # Jawline width
    jaw_width = distance.euclidean(jaw_points[3], jaw_points[13])
    
     # Cheekbone width
    cheekbone_width = distance.euclidean(
        (landmarks.part(1).x, landmarks.part(1).y),
        (landmarks.part(15).x, landmarks.part(15).y)
    )
    
     # Forehead width
    forehead_width = distance.euclidean(
        (landmarks.part(17).x, landmarks.part(17).y),
        (landmarks.part(26).x, landmarks.part(26).y)
    )
    
     # Chin shape (pointy or round)
    chin_pointiness = distance.euclidean(
        jaw_points[8],  # Bottom of chin
        np.mean([jaw_points[7], jaw_points[9]], axis=0)  # Points around chin
    )
    
     # Calculate ratios
    width_height_ratio = face_width / face_height if face_height > 0 else 0
    jaw_to_cheekbone_ratio = jaw_width / cheekbone_width if cheekbone_width > 0 else 0
    forehead_to_jaw_ratio = forehead_width / jaw_width if jaw_width > 0 else 0
    
    # Determine face shape
    if width_height_ratio > 0.75 and width_height_ratio < 0.85 and jaw_to_cheekbone_ratio < 0.9:
        return "Oval"
    elif width_height_ratio > 0.8 and width_height_ratio < 0.95 and jaw_to_cheekbone_ratio > 0.9:
        return "Round"
    elif jaw_to_cheekbone_ratio > 0.9 and forehead_to_jaw_ratio > 0.95 and forehead_to_jaw_ratio < 1.05:
        return "Square"
    elif width_height_ratio < 0.75 and jaw_to_cheekbone_ratio > 0.85:
        return "Rectangle"
    elif forehead_to_jaw_ratio > 1.1 and chin_pointiness > 10:
        return "Heart"
    elif cheekbone_width > face_width and chin_pointiness > 15:
        return "Diamond"
    else:
        return "Undetermined"
    
    def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    print("Face shape detection started. Press 'q' to quit.")
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            
            # Draw rectangle around face
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw dots for each facial landmark
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            # Determine face shape
            face_shape = calculate_face_shape(landmarks)
            
            # Display face shape
            cv2.putText(frame, f"Shape: {face_shape}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Face Shape Detection", frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Display the output
    cv2.imshow('Age and Gender Detection', frame)
    cv2.imshow("Face Shape Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()