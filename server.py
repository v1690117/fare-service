import logging
import os
import pickle
import random
import string

import face_recognition
import numpy as np
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

with open("model.pkl", "rb") as file:
    encodings = pickle.load(file)

app = Flask(__name__)

asgi_app = WsgiToAsgi(app)


def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


@app.route("/recognize", methods=["POST"])
def recognize():
    logger.info("process request received")
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]

    file_path = os.path.join(".", f"{generate_random_string(16)}-{file.filename}")
    file.save(file_path)
    logger.info("file saved")

    image = face_recognition.load_image_file(file_path)
    logger.info("image loaded")

    result = []
    face_encodings = face_recognition.face_encodings(image)
    logger.info("encodings calculated")

    # face_locations = face_recognition.face_locations(image)
    # logger.info("face locations calculated")

    for i in range(0, len(face_encodings)):
        percentage = np.mean(face_recognition.compare_faces(encodings, face_encodings[i]))
        # result.append((percentage, face_locations[i]))
        result.append((percentage))
    logger.info("encoding distance calculated")

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
