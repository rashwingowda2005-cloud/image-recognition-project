
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Image Recognition System",
    page_icon="📷",
    layout="centered"
)

st.title("AI Image Recognition System")
st.write("Upload an image to detect objects using Artificial Intelligence.")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Recognize Objects"):

        with st.spinner("Recognizing objects..."):

            results = model(image)

            detected_image = results[0].plot()

            st.subheader("Detection Results")

            st.image(
                detected_image,
                caption="Detected Objects",
                channels="BGR",
                use_container_width=True
            )

            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                st.write("Objects detected:")

                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    object_name = model.names[class_id]

                    st.write(
                        f"{object_name} - Confidence: {confidence * 100:.2f}%"
                    )
            else:
                st.warning("No objects detected.")
