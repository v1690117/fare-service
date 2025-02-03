import os
import pickle

import face_recognition

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

with open("model.pkl", "wb") as file:
    pickle.dump(encodings, file)
