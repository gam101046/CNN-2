from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
import shutil

# สร้างตัวแปรสำหรับการเพิ่มภาพ (Data Augmentation) โดยตั้งค่าให้หมุน ปรับขนาด ย้าย และทำการพลิก
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# กำหนด path สำหรับข้อมูลต้นฉบับและ path สำหรับเก็บข้อมูลที่ถูกเพิ่ม
base_path = '/Users/gam/Desktop/DEEP/CNN/DATASET'
output_path = '/Users/gam/Desktop/DEEP/CNN/DATASET/Augmented_dataset'
os.makedirs(output_path, exist_ok=True)  # สร้างโฟลเดอร์สำหรับเก็บข้อมูลที่ผ่านการเพิ่ม

max_images = 4000  # จำนวนภาพสูงสุดที่ต้องการในแต่ละหมวดหมู่

# กำหนดหมวดหมู่ของภาพและจำนวนภาพที่ต้องการในแต่ละหมวดหมู่
categories = {'Non Drowsy': 2000, 'Splitted_dataset': 0, 'Drowsy': 2000, 'Use mobile phone': 2000}

# วนลูปตามหมวดหมู่ต่าง ๆ
for category, total_images in categories.items():
    category_path = os.path.join(base_path, category)  # path ของภาพในแต่ละหมวดหมู่
    images = os.listdir(category_path)  # รายชื่อไฟล์ภาพในหมวดหมู่
    category_output_path = os.path.join(output_path, category)  # path สำหรับเก็บภาพที่ถูกเพิ่มในหมวดหมู่
    os.makedirs(category_output_path, exist_ok=True)  # สร้างโฟลเดอร์สำหรับเก็บภาพที่เพิ่มในหมวดหมู่นี้

    print(f"\n[INFO] Processing category: {category}")

    # คัดลอกภาพจากหมวดหมู่ต้นฉบับไปยังโฟลเดอร์ที่เก็บผลลัพธ์
    for img_name in images:
        src = os.path.join(category_path, img_name)  # path ของภาพต้นฉบับ
        dst = os.path.join(category_output_path, img_name)  # path ที่จะคัดลอกไป
        shutil.copy(src, dst)  # คัดลอกภาพ
        print(f"[COPY] Copied {img_name} to {category_output_path}")

    # ถ้าจำนวนภาพในหมวดหมู่ยังไม่ถึงจำนวนที่ต้องการให้ทำการเพิ่มภาพ (augmentation)
    if total_images < max_images:
        num_augments = max_images - total_images  # คำนวณจำนวนภาพที่ต้องการเพิ่ม
        print(f"[INFO] Starting augmentation for {category} to add {num_augments} images...")

        count = 0  # ตัวนับจำนวนภาพที่เพิ่ม
        for img_name in images:
            img_path = os.path.join(category_path, img_name)  # path ของภาพที่จะทำการเพิ่ม
            img = np.array(Image.open(img_path))  # เปิดภาพและแปลงเป็นอาเรย์

            # ถ้าภาพเป็นภาพขาวดำ (2 มิติ) ให้แปลงเป็น 3 มิติ (RGB)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)  # แปลงภาพเป็น 3 มิติ
                img = np.repeat(img, 3, axis=-1)  # ทำซ้ำค่าในแกนสี
            img = np.expand_dims(img, axis=0)  # เพิ่มมิติที่ 0 เพื่อให้ตรงกับการรับข้อมูลของ DataGenerator

            # สร้างภาพใหม่โดยใช้ Data Augmentation และบันทึกไปที่โฟลเดอร์ output
            for batch in datagen.flow(img, batch_size=1, save_to_dir=category_output_path, save_prefix='aug', save_format='jpeg'):
                count += 1  # เพิ่มตัวนับ
                aug_img_name = f"aug_{count}_{img_name}"  # ตั้งชื่อไฟล์ภาพที่ถูกเพิ่ม
                print(f"[AUGMENT] Generated {aug_img_name} for {category}")
                if count >= num_augments:  # ถ้าครบจำนวนภาพที่ต้องการเพิ่มแล้ว ให้หยุด
                    break

            if count >= num_augments:  # ถ้าครบจำนวนภาพที่ต้องการเพิ่มแล้ว ให้หยุด
                break

        print(f"[INFO] Finished augmentation for {category}: {count} augmented images generated.")
    else:
        print(f"[INFO] No augmentation needed for {category} as it already has {total_images} images.")

# แสดงข้อความเมื่อกระบวนการทั้งหมดเสร็จสิ้น
print("[INFO] Augmentation process complete!")
