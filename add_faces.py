import cv2
import pickle
import numpy as np
import os

# Create the 'data' directory if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("Enter your Aadhar number: ")

framesTotal = 51
captureAfterFrame = 2

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < framesTotal and i % captureAfterFrame == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= framesTotal:
        break

video.release()
cv2.destroyAllWindows()

# Convert face data to NumPy array and reshape
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape((framesTotal, -1))

# Save names data
names_file = 'data/names.pkl'
faces_file = 'data/faces_data.pkl'

if not os.path.exists(names_file):
    names = [name] * framesTotal
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * framesTotal)
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# Save faces data
if os.path.exists(faces_file):
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)  # Load existing data
else:
    faces = np.empty((0, faces_data.shape[1]))  # Initialize empty array if file does not exist

faces = np.vstack((faces, faces_data))  # Stack new face data
with open(faces_file, 'wb') as f:
    pickle.dump(faces, f)

print("Data saved successfully!")

