from ultralytics import YOLO

model = YOLO("yolov12n.pt")

dataset_loaction = "/home/hasyim/Bootcamp_AI/modul4/hari6-datecttion/data_plate/content/License-Plate-Recognition-11"

results = model.train(
    data = f"{dataset_loaction}/data.yaml",
    epochs=20,
    batch=16,
    imgsz=640,
    exist_ok=True,
    patience=5,
    save_period=5,
    val=True,
    verbose=True,
    flipud=0.5
)