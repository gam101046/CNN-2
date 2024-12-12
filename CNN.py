from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# โหลดโมเดล CNN
model = tf.keras.models.load_model("path_to_your_model.h5")

# ฟังก์ชันสำหรับทำนายการง่วง
def predict_drowsiness(frame):
    resized_frame = cv2.resize(frame, (64, 64))  # ขนาดภาพที่โมเดลรองรับ
    normalized_frame = resized_frame / 255.0  # ปรับค่าสีให้อยู่ในช่วง [0,1]
    input_data = np.expand_dims(normalized_frame, axis=0)  # เพิ่มมิติ
    prediction = model.predict(input_data)
    return prediction[0][0]  # ผลลัพธ์ของโมเดล

# ฟีดกล้อง
def gen_frames():
    camera = cv2.VideoCapture(0)  # เปิดกล้อง
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # เพิ่มกล่องข้อความแสดงสถานะการง่วง
            prediction = predict_drowsiness(frame)
            status = "Drowsy" if prediction > 0.5 else "Alert"
            cv2.putText(frame, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ส่งเฟรมไปยัง frontend
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
