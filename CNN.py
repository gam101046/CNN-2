from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# โหลดโมเดล CNN ที่ได้รับการฝึกใหม่
model = tf.keras.models.load_model("/Users/gam/Desktop/DEEP/CNN/cnn_model.h5")  # ใช้เส้นทางของโมเดลใหม่

# ฟังก์ชันสำหรับทำนายสถานะ
def predict_status(frame):
    resized_frame = cv2.resize(frame, (128, 128))  # ขนาดภาพที่โมเดลรองรับ
    normalized_frame = resized_frame / 255.0  # ปรับค่าสีให้อยู่ในช่วง [0,1]
    input_data = np.expand_dims(normalized_frame, axis=0)  # เพิ่มมิติ
    prediction = model.predict(input_data)

    # ทำนายผลลัพธ์จากโมเดล (ค่าผลลัพธ์ของโมเดลมี 3 คลาส: Alert, Drowsy, Mobile)
    if prediction[0][0] > 0.5:
        return "Alert"
    elif prediction[0][1] > 0.5:
        return "Drowsy"
    else:
        return "Mobile"

# ฟังก์ชันสำหรับส่งข้อมูลภาพจากกล้อง
def gen_frames():
    camera = cv2.VideoCapture(0)  # เปิดกล้อง
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # ทำนายสถานะ
            status = predict_status(frame)
            
            # เพิ่มกล่องข้อความแสดงสถานะ
            if status == "Alert":
                cv2.putText(frame, "Status: Alert", (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
            elif status == "Drowsy":
                cv2.putText(frame, "Status: Drowsy", (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Status: Mobile", (50, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 3)

            # ส่งเฟรมไปยัง frontend
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # หน้า HTML ที่จะแสดงผล

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # เปิดให้เข้าถึงจากทุกอุปกรณ์ในเครือข่ายเดียวกัน
