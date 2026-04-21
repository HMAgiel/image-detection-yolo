````markdown
# 👁️ Dual-Task Vision System: Quality Assurance & Plate Recognition

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)]()
[![YOLO](https://img.shields.io/badge/YOLO-v12-yellow?logo=ultralytics&logoColor=black)]()
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv&logoColor=white)]()

A modular computer vision project demonstrating object detection applied to two distinct industry use cases: **Automated License Plate Recognition (ALPR)** and **Industrial Glass Defect Detection**. 

This repository leverages state-of-the-art YOLOv12 models wrapped in an interactive web interface using Gradio, showcasing end-to-end capabilities from model training to inference deployment.

---

## 🚀 Key Features

* **🏭 Manufacturing QA (Glass Defect Detection):**
    * Identifies and localizes defects in glass products.
    * Provides real-time counting of identified objects (glass vs. defect).
    * Outputs conditional visual HTML alerts to immediately notify inspectors of failed quality checks.
* **🚗 Smart City / Logistics (License Plate Detection):**
    * Detects vehicle license plates with high accuracy.
    * Automatically extracts and crops the plate region for downstream OCR processing.
* **💻 Interactive Web UI:** Built with Gradio to provide an intuitive, drag-and-drop interface for users to test images instantly.
* **⚙️ Optimized Inference:** Utilizes OpenCV and Numpy for efficient bounding box rendering and array slicing.

---

## 📂 Project Architecture

The project is structured into modular components separating the UI, inference logic, and training pipeline:

| File | Description |
| :--- | :--- |
| `detection.py` | The core inference engine. Loads the fine-tuned YOLO weights and handles image processing, thresholding, and dynamic bounding box generation. |
| `main_glass.py` | The Gradio application script for the **Glass Defect** interface. |
| `main_plate.py` | The Gradio application script for the **License Plate** interface. |
| `train.py` | The training pipeline script utilizing `ultralytics` to fine-tune YOLOv12n with custom hyperparameter configurations. |

---

## 🛠️ Installation & Setup

To get this project running locally, it is recommended to use a fast virtual environment manager like `uv` or standard `pip`.

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name
````

**2. Set up the environment**

```bash
# Using uv for faster dependency resolution
uv venv
source .venv/bin/activate

# Or using standard pip
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install ultralytics gradio opencv-python-headless pillow numpy
```

**4. Ensure your model weights are in place**
Make sure `best_glass.pt` and `best_license.pt` are located in the appropriate directory as referenced in `detection.py`.

-----

## 🎮 Usage

You can launch either of the detection systems independently based on the task you want to test.

\<details\>
\<summary\>\<b\>🔍 Launch the Glass Defect Detection System\</b\>\</summary\>

Run the following command to start the industrial QA interface:

```bash
python main_glass.py
```

*Open the provided local URL (usually `http://127.0.0.1:7860`) in your browser. Upload an image of a glass product to see the bounding boxes and inspection status.*

\</details\>

\<details\>
\<summary\>\<b\>🚙 Launch the License Plate Detection System\</b\>\</summary\>

Run the following command to start the ALPR interface:

```bash
python main_plate.py
```

*Upload a car image. The system will output the full annotated image alongside a cropped version of the license plate.*

\</details\>

-----

## 🧠 Model Training Details

The models were trained using the `ultralytics` framework. The training script (`train.py`) demonstrates custom configurations to optimize the model:

  * **Base Model:** `yolov12n.pt` (Nano architecture for fast inference speed)
  * **Image Size:** 640x640
  * **Batch Size:** 16
  * **Epochs:** 20 (with an early stopping patience of 5 to prevent overfitting)
  * **Augmentation:** Vertical flipping (`flipud=0.5`) utilized to improve generalization.

-----