import cv2
import pickle
from skimage.transform import resize
import numpy as np

# Load the trained model
model = pickle.load(open('./model.p', 'rb'))

# Load video
video_path = r'C:\Users\waghd\Downloads\data-20250704T174533Z-1-001\data\parking_1920_1080_loop.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and reshape for prediction
    resized = resize(frame, (15, 15))
    flat_input = resized.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(flat_input)
    label = "Not Empty" if prediction[0] == 1 else "Empty"

    # Display
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
