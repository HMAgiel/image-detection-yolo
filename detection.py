from ultralytics import YOLO
from collections import Counter
from PIL import Image
import cv2
import numpy as np

MODEL = YOLO("/home/hasyim/Bootcamp_AI/modul4/hari6-datecttion/best_glass.pt")
MODEL_LINCESE = YOLO("/home/hasyim/Bootcamp_AI/modul4/hari6-datecttion/best_license.pt")
CLASS_NAME = ["defect", "glass"]
CLASS_NAME_PLATE = ["plate"]

def detect_defect(image):
    results = MODEL.predict(source=image, save=False, conf=0.3)[0]
    image_np = np.array(image)
    annotated_img = image_np.copy()
    
    detections = []
    class_counter = Counter()
    defect_found = False
    
    for box in results.boxes:
        cls_id = int(box.cls)
        label = CLASS_NAME[cls_id]
        conf = float(box.conf)
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        
        class_counter[label] += 1
        
        if label == "defect":
            defect_found = True
            
        color = (0, 255, 0) if label == "glass" else (255, 0, 0)
        cv2.rectangle(annotated_img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated_img, f"{label} [{conf:.2f}]", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        summary = "\n".join([f"{label}: {count}" for label, count in class_counter.items()])
        
        if defect_found:
            status_html = (
                "<div style='padding:10px; background-color:#ffe5e5; color:#a30000;"
                "border-left: 6px solid #0f0; font-weight:bold;'>DEFECT DETECTED! PLEASE INSPECT!.</div>"
            )
        else:
            status_html = (
                "<div style='padding:10px; background-color:#e6ffe6; color:#006600;"
                "border-left: 6px solid #0f0; font-weight:bold;'>GA ADA DEFECT! YEAY!.</div>"
            )
        
    return Image.fromarray(annotated_img), summary or "NO OBJECT DETECTED!", status_html

def detect_license(image):
    results = MODEL_LINCESE.predict(source=image, save=False, conf=0.3)[0]
    image_np = np.array(image)
    annotated_img = image_np.copy()
    
    plate_detected = False
    cropped_plate = None # Siapkan variabel penampung gambar crop
    
    for box in results.boxes:
        cls_id = int(box.cls)
        label = CLASS_NAME_PLATE[cls_id]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if label == "plate":
            plate_detected = True
            
            # --- PROSES CROP DILAKUKAN DI SINI ---
            # Jika ada lebih dari 1 plat, kode ini hanya akan mengambil plat pertama yang terdeteksi
            if cropped_plate is None: 
                # Crop menggunakan Numpy Slicing: [y_awal:y_akhir, x_awal:x_akhir]
                cropped = image_np[y1:y2, x1:x2]
                
        # Tentukan warna kotak (Hijau untuk plate, Biru/Merah untuk lainnya)
        color = (0, 255, 0) if label == "plate" else (255, 0, 0) 
        
        # Gambar kotak dan teks pada gambar utama
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, f"{label} [{conf:.2f}]", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Logika Status HTML
    if plate_detected:
        status_html = (
            "<div style='padding:10px; background-color:#e6ffe6; color:#006600;"
            "border-left: 6px solid #0f0; font-weight:bold;'>PLATE DETECTED!</div>"
        )
    else:
        status_html = (
            "<div style='padding:10px; background-color:#ffe5e5; color:#a30000;"
            "border-left: 6px solid #f00; font-weight:bold;'>NO PLATE!</div>"
        )
        
    # Kembalikan: Gambar Full berr-annotasi, Gambar Plat Hasil Crop, Status HTML
    return Image.fromarray(annotated_img), Image.fromarray(cropped), status_html