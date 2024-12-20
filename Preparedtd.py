import os
from sklearn.model_selection import train_test_split
import shutil

# กำหนดเส้นทางของชุดข้อมูล
base_path = '/Users/gam/Desktop/DEEP/CNN/DATASET'

# ฟังก์ชั่นตรวจสอบว่าไฟล์เป็นรูปภาพหรือไม่
def is_image(file_name):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    return any(file_name.endswith(ext) for ext in valid_extensions)

# นับจำนวนรูปภาพในแต่ละหมวดหมู่
categories = {}
for category in os.listdir(base_path):
    category_path = os.path.join(base_path, category)
    if os.path.isdir(category_path):
        categories[category] = len([f for f in os.listdir(category_path) if is_image(f)])

# หาจำนวนสูงสุดของรูปภาพในแต่ละหมวดหมู่
max_images = max(categories.values())
print("Number of images in each class:", categories)
print("All classes will be augmented to:", max_images)

# กำหนดเส้นทางของชุดข้อมูลที่ได้จากการเพิ่มรูปภาพ (Augmented dataset)
augmented_dataset_path = '/Users/gam/Desktop/DEEP/CNN/DATASET/Augmented_dataset'  # เส้นทางไปยังชุดข้อมูลที่เพิ่มรูปภาพแล้ว
output_path = '/Users/gam/Desktop/DEEP/CNN/DATASET/Splitted_dataset'  # เส้นทางที่เก็บชุดข้อมูลที่แบ่งแล้ว
os.makedirs(output_path, exist_ok=True)

# ดึงรายชื่อหมวดหมู่ทั้งหมดจากโฟลเดอร์ชุดข้อมูล
categories = [category for category in os.listdir(augmented_dataset_path) if os.path.isdir(os.path.join(augmented_dataset_path, category))]

# สร้างโฟลเดอร์สำหรับ train, validation, และ test
split_dirs = ['train', 'val', 'test']
for split in split_dirs:
    for category in categories:
        os.makedirs(os.path.join(output_path, split, category), exist_ok=True)

# แบ่งชุดข้อมูลเป็น train, val, และ test
print("[INFO] Starting dataset splitting...")
for category in categories:
    category_path = os.path.join(augmented_dataset_path, category)
    images = [img for img in os.listdir(category_path) if is_image(img)]  # เลือกเฉพาะไฟล์รูปภาพ

    # ตรวจสอบว่ามีรูปภาพในหมวดหมู่นี้หรือไม่
    if len(images) == 0:
        print(f"[WARNING] No images found in category: {category}. Skipping...")
        continue

    # แสดงข้อมูลเกี่ยวกับหมวดหมู่ที่กำลังประมวลผล
    print(f"\n[INFO] Processing category: {category}")
    print(f"[INFO] Total images in {category}: {len(images)}")

    # แบ่งรูปภาพเป็น 70% train, 20% validation, 10% test
    train_images, temp_images = train_test_split(images, test_size=(1 - 0.7), random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=(0.1 / (0.2 + 0.1)), random_state=42)

    # แสดงจำนวนรูปภาพในแต่ละชุด
    print(f"[INFO] {len(train_images)} images for training.")
    print(f"[INFO] {len(val_images)} images for validation.")
    print(f"[INFO] {len(test_images)} images for testing.")

    # คัดลอกรูปภาพไปยังโฟลเดอร์ที่กำหนด
    print(f"[INFO] Copying images for {category}...")
    for img in train_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(output_path, 'train', category, img)
        shutil.copy(src, dst)
        print(f"[COPY] {img} copied to train/{category}")

    for img in val_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(output_path, 'val', category, img)
        shutil.copy(src, dst)
        print(f"[COPY] {img} copied to val/{category}")

    for img in test_images:
        src = os.path.join(category_path, img)
        dst = os.path.join(output_path, 'test', category, img)
        shutil.copy(src, dst)
        print(f"[COPY] {img} copied to test/{category}")

print("\n[INFO] Dataset splitting complete!")
print(f"[INFO] Splitted dataset is saved in: {output_path}")




# import tensorflow as tf
# import cv2
# import numpy as np

# # โหลดโมเดล CNN ที่ฝึกมาแล้ว
# model = tf.keras.models.load_model("/Users/gam/Desktop/DEEP/CNN/cnn_model.h5")

# # ฟังก์ชันสำหรับทำนายสถานะจากรูปภาพ
# def predict_status(image_path):
#     # อ่านรูปภาพจากไฟล์
#     image = cv2.imread(image_path)
    
#     # เปลี่ยนขนาดของรูปภาพให้ตรงกับที่โมเดลต้องการ (128x128)
#     resized_image = cv2.resize(image, (128, 128))  # ขนาดที่โมเดลรองรับ
#     normalized_image = resized_image / 255.0  # ปรับค่าสีให้อยู่ในช่วง [0,1]
    
#     # เพิ่มมิติให้กับข้อมูล (เพื่อให้เข้ากับอินพุตของโมเดล)
#     input_data = np.expand_dims(normalized_image, axis=0)  # เพิ่มมิติ
#     prediction = model.predict(input_data)
#     print(prediction)
    
#     # ทำนายสถานะ (คาดว่าโมเดลมี 3 คลาส: Alert, Drowsy, Mobile)
#     if prediction[0][0] > 0.5:
#         return "Drowsy"
#     elif prediction[0][1] > 0.5:
#         return "Non Drowsy"
#     else:
#         return "Use Mobile phone"

# # ระบุเส้นทางของรูปภาพที่ต้องการทดสอบ
# image_path = "/Users/gam/Desktop/aug_0_187.jpeg"

# # ทำนายสถานะจากรูปภาพ
# status = predict_status(image_path)
# print(f"Prediction for the image: {status}")

