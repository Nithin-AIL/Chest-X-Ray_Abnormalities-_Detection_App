# CliniScan – Chest X‑Ray Abnormalities Detection

**Live App:** [https://chest-x-rayabnormalities-detectionapp-ngebmiebclx8e8xjbb9f8v.streamlit.app/](https://chest-x-rayabnormalities-detectionapp-ngebmiebclx8e8xjbb9f8v.streamlit.app/)

CliniScan is a web‑based medical imaging assistive tool that performs **multi‑label disease classification**, **YOLOv8 abnormality detection**, and **Grad‑CAM visual explanation** on chest X‑ray images.

It combines:

* **EfficientNet‑B4** (via `timm`) for classification
* **YOLOv8** for localization & detection
* **Grad‑CAM** for heatmap visualization of model attention
* **Streamlit** for fast and interactive web UI

---

##  Features

### Multi‑Label Disease Classification

Predicts 15 chest‑related abnormalities including:

* Aortic Enlargement
* Atelectasis
* Cardiomegaly
* Consolidation
* ILD
* Infiltration
* Lung Opacity
* Nodule/Mass
* Pleural Effusion
* Pneumothorax
* Pulmonary Fibrosis
* …and more

###  Grad‑CAM Visualization

Highlights the hottest regions influencing the classifier’s decision.

###  YOLOv8 Detection

Performs bounding‑box detection for visible abnormalities (except "No finding").

###  Clean & Responsive UI

Modern Streamlit UI with:

* Sidebar actions
* Adjustable image preview size
* Grad‑CAM intensity slider
* Detection overlay visualization

---

##  Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── uploaded_models/
│   ├── classification_model.pth
│   └── detection_model.pt
├── sample_images/         # Example chest X‑ray images
└── .streamlit/
    └── runtime.txt        # Python version fix for Streamlit Cloud
```

---

##  Installation (Local)

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
pip install -r requirements.txt
streamlit run app.py
```

---

##  Deployment (Streamlit Cloud)

This app uses:

* `.streamlit/runtime.txt` → to force Python **3.10** (fixes OpenCV & NumPy issues)
* `opencv-python-headless` for server‑friendly CV

### Required Files in Repo

✔ `app.py`
✔ `requirements.txt`
✔ `.streamlit/runtime.txt`
✔ `uploaded_models/` containing both models
✔ `sample_images/`

Once pushed to GitHub → deploy at: [https://share.streamlit.io](https://share.streamlit.io)

---

##  Requirements

```
streamlit
torch
torchvision
ultralytics
opencv-python-headless==4.7.0.72
pillow
numpy==1.26.4
matplotlib
timm
scipy
```

 `numpy==1.26.4` is important to avoid ABI mismatch with OpenCV & PyTorch.

---

##  Models

* **EfficientNet‑B4** classifier (`classification_model.pth`)
* **YOLOv8** detector (`detection_model.pt`)

Place them inside:

```
uploaded_models/
```

---

##  Usage

1. Upload a chest X‑ray (`.jpg` / `.png`).
2. Run **Classification**, **Grad‑CAM**, or **YOLO Detection** from the sidebar.
3. View predictions, heatmaps, and bounding boxes.

---

##  Contact

Maintainer: **Nithin** (Nithin‑ARK)

Feel free to open issues or request improvements.

---

## ⭐ Support the Project

If you like this project, consider giving it a ⭐ on GitHub!
