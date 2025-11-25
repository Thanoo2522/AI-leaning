import os
import json
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow.lite as tflite
import firebase_admin
from firebase_admin import credentials, storage

# -------------------------------
# 1) Firebase Init
# -------------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

service_account_json = os.environ.get("FIREBASE_SERVICE_KEY")
if not service_account_json:
    raise Exception("Environment variable FIREBASE_SERVICE_KEY not found!")

cred_dict = json.loads(service_account_json)
cred = credentials.Certificate(cred_dict)

firebase_admin.initialize_app(cred, {
    "storageBucket": "lotteryview.firebasestorage.app"
})

bucket = storage.bucket()

# -------------------------------
# 2) โหลด TFLite model จาก Firebase Storage
# -------------------------------
print("Downloading model from Firebase...")

local_path = os.path.join(UPLOAD_FOLDER, "model.tflite")
blob = bucket.blob("test_AI_Leaning_kit/model.tflite")
blob.download_to_filename(local_path)

print(f"Model downloaded → {local_path}")

# -------------------------------
# 3) Load TFLite Model (load once)
# -------------------------------
interpreter = tflite.Interpreter(model_path=local_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mapping index → class name
CLASS_NAMES = ["red", "yellow", "pink"]

# -------------------------------
# Flask Server
# -------------------------------
app = Flask(__name__)

@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    img = Image.open(file.stream).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0]  # shape = (3,)

    # หาค่า score ที่มากที่สุด
    max_index = int(np.argmax(prediction))
    color = CLASS_NAMES[max_index]

    return jsonify({
        "color": color,
        "scores": {
            "red": float(prediction[0]),
            "yellow": float(prediction[1]),
            "pink": float(prediction[2])
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
