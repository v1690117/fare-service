import os

import face_recognition
import numpy as np
import pickle

with open("model.pkl", "rb") as file:
    encodings = pickle.load(file)

def matches(img):
    fn = face_recognition.load_image_file(img)
    fn_encoding = face_recognition.face_encodings(fn)[0]
    results = face_recognition.compare_faces(encodings, fn_encoding)
    return np.mean(results)


test_data_dir = "data/tests"
test_files = [f for f in os.listdir(test_data_dir) if os.path.isfile(os.path.join(test_data_dir, f))]
for file in test_files:
    percentage = matches(os.path.join(test_data_dir, file))
    print(f"{file}: {percentage}")
