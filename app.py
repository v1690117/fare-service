import os

import face_recognition
import numpy as np

folder_path = "data/samples"
encodings = []

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for file in files:
    try:
        image = face_recognition.load_image_file(os.path.join(folder_path, file))
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) > 0:
            encodings.append(face_encodings[0])
            print(f"file {file} processed successfully.")
        else:
            print(f"no face found at {file}.")
    except Exception as e:
        print(f"error on processing {file}: {e}")

print("-----------------------------------")
print(f"total calculated: {len(encodings)}")
print()


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
