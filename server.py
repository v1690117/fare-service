import pickle

import face_recognition
import numpy as np
from flask import Flask, request, jsonify

with open("model.pkl", "rb") as file:
    encodings = pickle.load(file)

app = Flask(__name__)


@app.route("/recognize", methods=["POST"])
def recognize():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    image = face_recognition.load_image_file(file)

    result = []
    face_encodings = face_recognition.face_encodings(image)
    face_locations = face_recognition.face_locations(image)

    for i in range(0, len(face_encodings)):
        percentage = np.mean(face_recognition.compare_faces(encodings, face_encodings[i]))
        result.append((percentage, face_locations[i]))

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
