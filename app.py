# conda activate yolo_env
#  cd C:\Users\S G Patil\yolov8_pallet_app
#streamlit run app.py






import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import tempfile

# Load the trained model
model = YOLO('C:/Users/S G Patil/yolov8_pallet_app/best.pt')

st.title("Pallets Detection and Counting-263")
st.write("Upload an image to detect objects using your custom-trained model.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)  # run YOLO on the saved image

    # Plot and show detection result
    result_img = results[0].plot()
    st.image(result_img, caption="Detection Result", use_container_width=True)

    # Count detected objects
    detected_classes = results[0].boxes.cls.cpu().numpy()  # class indices
    class_names = results[0].names                     # index-to-name mapping
    class_counts = {}

    for cls_idx in detected_classes:
        class_name = class_names[int(cls_idx)]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    st.markdown("### Detection Summary:")
    for class_name, count in class_counts.items():
        st.write(f"- **{class_name}**: {count} detected")

    # Optional: Show only pallets
    if "pallet" in class_counts:
        st.success(f"✅ Total pallets detected: {class_counts['pallet']}")
    else:
        st.warning("⚠️ No pallets detected.")