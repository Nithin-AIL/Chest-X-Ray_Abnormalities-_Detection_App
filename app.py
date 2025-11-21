
# FINAL app.py — improved Grad-CAM processing only (UI & sizes unchanged)
import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from ultralytics import YOLO
import timm
import os
import scipy.ndimage

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="CliniScan", layout="wide")

# Clean instruction card
st.markdown("""
<div style="text-align:center; padding: 10px 0 10px 0;">
    <h1 style="color:#2C3E50; margin:0;"> CliniScan</h1>
    <p style="color:#7F8C8D; margin:4px 0 12px 0;">Lung Abnormality Detection — EfficientNet-B4 (timm) + YOLOv8</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background-color:#BF77F6;
    padding:14px;
    border-radius:10px;
    border:1px solid #E5E7EB;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    font-size:15px;
">
<b>📘 Quick Instructions</b>

<ol style="margin-top:10px; padding-left:20px; line-height:1.6;">

  <li>
    Upload a chest X-ray image (JPG or PNG).
  </li>

  <li>
    Use the left sidebar to run the desired actions:
    <ul style="margin-top:6px; padding-left:20px; line-height:1.6;">
      <li><b>Run Classification</b></li>
      <li><b>Show Grad-CAM</b></li>
      <li><b>Run YOLO Detection</b></li>
    </ul>
  </li>

  <li>
    Results are displayed based on the selected option.
  </li>

</ol>

</div>
""", unsafe_allow_html=True)

# -------------------------
# Paths & labels
# -------------------------
MODEL_DIR = "uploaded_models"
CLASS_MODEL = os.path.join(MODEL_DIR, "classification_model.pth")
DET_MODEL = os.path.join(MODEL_DIR, "detection_model.pt")

class_names = [
    'Aortic enlargement','Atelectasis','Calcification','Cardiomegaly',
    'Consolidation','ILD','Infiltration','Lung Opacity','No finding',
    'Nodule/Mass','Other lesion','Pleural effusion','Pleural thickening',
    'Pneumothorax','Pulmonary fibrosis'
]

# -------------------------
# Utils: find last conv layer
# -------------------------
def find_last_conv_layer(model):
    from torch.nn import Conv2d
    last = None
    for _, m in model.named_modules():
        if isinstance(m, Conv2d):
            last = m
    return last

# -------------------------
# GradCAM class
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target = target_layer
        self.grad = None
        self.act = None

        def forward_hook(m, inp, outp):
            self.act = outp.detach()

        def backward_hook(m, gin, gout):
            self.grad = gout[0].detach()

        self.target.register_forward_hook(forward_hook)
        try:
            self.target.register_full_backward_hook(backward_hook)
        except Exception:
            self.target.register_backward_hook(backward_hook)

    def __call__(self, x, cls_idx):
        x = x.to("cpu")
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, cls_idx].backward()

        g = self.grad
        a = self.act
        w = g.mean((2,3), keepdim=True)
        cam = (w * a).sum(1).squeeze(0).cpu().numpy()
        cam = np.maximum(cam, 0)
        return cam

# -------------------------
# Load models
# -------------------------
@st.cache_resource
def load_clf():
    import torch.serialization
    import numpy as np
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

    ck = torch.load(CLASS_MODEL, map_location="cpu", weights_only=False)
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(class_names))

    if isinstance(ck, dict) and "model_state" in ck:
        cleaned_sd = {k.replace("backbone.", ""): v for k, v in ck["model_state"].items()}
    elif isinstance(ck, dict) and "state_dict" in ck:
        cleaned_sd = {k.replace("module.", "").replace("backbone.", ""): v for k,v in ck["state_dict"].items()}
    else:
        cleaned_sd = {k.replace("module.", "").replace("backbone.", ""): v for k,v in ck.items()}

    model.load_state_dict(cleaned_sd, strict=False)
    thresholds = ck.get("thresholds", None)
    return model.eval(), thresholds

@st.cache_resource
def load_det():
    return YOLO(DET_MODEL)

clf, saved_thresholds = load_clf()
det = load_det()
last_conv = find_last_conv_layer(clf)

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("⚙ Actions")
    run_class_btn = st.button(" Run Classification")
    run_gradcam_btn = st.button(" Show Grad-CAM")
    run_yolo_btn = st.button(" Run YOLO Detection")
    st.markdown("---")
    grad_alpha = st.slider("Grad-CAM intensity (alpha heatmap)", 0.0, 1.0, 0.3, 0.05)
    max_display_width = st.number_input("Display image width (px)", min_value=300, max_value=1200, value=600, step=50)

# -------------------------
# Session state initialization
# -------------------------
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "class_probs" not in st.session_state:
    st.session_state.class_probs = None

if "class_results" not in st.session_state:
    st.session_state.class_results = None

if "gradcam_heat" not in st.session_state:
    st.session_state.gradcam_heat = None
if "gradcam_overlay" not in st.session_state:
    st.session_state.gradcam_overlay = None

if "det_img" not in st.session_state:
    st.session_state.det_img = None
    st.session_state.det_abn = None

# -------------------------
# Uploader
# -------------------------
uploaded = st.file_uploader("Upload Chest X-ray (JPG / PNG)", type=["jpg","jpeg","png"])

if uploaded:
    if st.session_state.uploaded_name != uploaded.name:
        st.session_state.uploaded_name = uploaded.name
        st.session_state.class_probs = None
        st.session_state.class_results = None
        st.session_state.gradcam_heat = None
        st.session_state.gradcam_overlay = None
        st.session_state.det_img = None
        st.session_state.det_abn = None

    img = Image.open(uploaded).convert("RGB")
    preview_w = min(max_display_width, img.size[0])
    preview_h = int(preview_w * img.size[1] / img.size[0])
    st.image(img.resize((preview_w, preview_h)), caption="Uploaded X-ray (preview)", width=preview_w)

    tf = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    x = tf(img).unsqueeze(0)

    # -------------------------
    # Classification
    # -------------------------
    if run_class_btn:
        with st.spinner("Running multi-label classification..."):
            logits = clf(x)
            probs = torch.sigmoid(logits)[0].detach()
            st.session_state.class_probs = probs

            if saved_thresholds is not None:
                try:
                    thr = torch.tensor(saved_thresholds, dtype=probs.dtype)
                except:
                    thr = torch.full_like(probs, 0.5)
            else:
                thr = torch.full_like(probs, 0.5)

            preds = (probs > thr).int()
            results = [(class_names[i], float(probs[i])) for i in range(len(class_names)) if preds[i] == 1]
            st.session_state.class_results = results

    if st.session_state.class_results is not None:
        st.subheader("🟩 Classification (multi-label)")
        if len(st.session_state.class_results) == 0:
            st.info("No abnormalities predicted by classifier.")
        else:
            for name, p in st.session_state.class_results:
                st.write(f"- **{name}** — {p:.3f}")

    # -------------------------
    # Grad-CAM
    # -------------------------
    if run_gradcam_btn:
        if st.session_state.class_probs is None:
            st.warning("Please run Classification first.")
        else:
            with st.spinner("Generating Grad-CAM..."):
                probs = st.session_state.class_probs
                top_idx = int(probs.argmax().item())
                top_class = class_names[top_idx]

                if top_class == "No finding":
                    st.info("Grad-CAM disabled because classifier predicted **No finding**.")
                    st.session_state.gradcam_heat = None
                    st.session_state.gradcam_overlay = None
                else:
                    cam_gen = GradCAM(clf, last_conv)
                    cam = cam_gen(x, top_idx)

                    orig_w, orig_h = img.size
                    cam_resized = cv2.resize(cam, (orig_w, orig_h))

                    lower_pct = np.percentile(cam_resized, 20)
                    cam_clip = cam_resized - lower_pct
                    cam_clip[cam_clip < 0] = 0.0

                    cam_norm = cam_clip / (cam_clip.max() + 1e-8)
                    cam_norm = cam_norm ** 0.6
                    cam_smooth = scipy.ndimage.gaussian_filter(cam_norm, sigma=1.2)

                    heat = np.uint8(255 * cam_smooth)
                    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

                    alpha = float(grad_alpha)
                    alpha = max(0.25, min(0.6, alpha))

                    base = np.array(img)
                    overlay = cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0)

                    st.session_state.gradcam_heat = heat
                    st.session_state.gradcam_overlay = overlay

    if st.session_state.gradcam_overlay is not None:
        st.subheader(" Grad-CAM")
        cols = st.columns(2)
        cols[0].image(st.session_state.gradcam_heat, caption="Heatmap (full image)", width=min(max_display_width, img.size[0]))
        cols[1].image(st.session_state.gradcam_overlay, caption="Overlay (full image)", width=min(max_display_width, img.size[0]))

    # -------------------------
    # YOLO detection  (ONLY FIXED PART)
    # -------------------------
    if run_yolo_btn:
        with st.spinner("Running YOLOv8 detection..."):
            results = det.predict(np.array(img), conf=0.25)
            boxes = results[0].boxes

            det_img = np.array(img).copy()
            abnormalities = []

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy_np = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy_np.tolist()
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])

                    label_name = class_names[cls_idx]

                    #  Skip "No finding"
                    if label_name == "No finding":
                        continue

                    # ✔ Draw for abnormalities only
                    cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{label_name} ({conf:.2f})"
                    abnormalities.append(label)

                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    tx = x1
                    ty = y1 - 8

                    if ty - th - baseline < 0:
                        ty = y1 + th + 8
                    if tx + tw > det_img.shape[1] - 4:
                        tx = max(4, det_img.shape[1] - tw - 4)

                    cv2.rectangle(det_img, (tx - 2, ty - th - baseline - 2),
                                   (tx + tw + 2, ty + 2), (0, 255, 0), -1)
                    cv2.putText(det_img, label, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            st.session_state.det_img = det_img
            st.session_state.det_abn = abnormalities

    if st.session_state.det_img is not None:
        st.subheader(" YOLOv8 Detection")
        st.image(st.session_state.det_img, caption="Detections", width=min(max_display_width, img.size[0]))
        if st.session_state.det_abn:
            st.error("Detected:")
            for a in st.session_state.det_abn:
                st.write("- ", a)
        else:
            st.success("No abnormalities detected.")


