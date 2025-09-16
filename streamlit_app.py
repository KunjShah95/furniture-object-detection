# streamlit_app.py
import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = st.secrets.get(
    "api_url", "http://localhost:8000/predict"
)  # set via Streamlit secrets or change

st.set_page_config(page_title="Furniture Detector", layout="centered")

st.title("Furniture Detector — Upload room photo")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
mask_objects = st.checkbox("Generate object masks", value=False)
remove_objects = st.checkbox("Remove detected objects from image", value=False)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    if st.button("Detect"):
        with st.spinner("Uploading and detecting..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }
            params = {
                "conf_threshold": conf,
                "mask_objects": mask_objects,
                "remove_objects": remove_objects,
            }
            try:
                resp = requests.post(API_URL, files=files, params=params, timeout=60)
                resp.raise_for_status()
            except Exception as e:
                st.error(f"API request failed: {e}")
            else:
                data = resp.json()
                # annotated image
                annotated_b64 = data.get("annotated_image_base64")
                if annotated_b64:
                    annotated_bytes = base64.b64decode(annotated_b64)
                    annotated_img = Image.open(BytesIO(annotated_bytes))
                    st.image(annotated_img, caption="Detections", use_column_width=True)

                # mask image
                mask_b64 = data.get("mask_image_base64")
                if mask_b64:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_img = Image.open(BytesIO(mask_bytes))
                    st.image(mask_img, caption="Object Masks", use_column_width=True)

                boxes = data.get("boxes", [])
                if boxes:
                    st.subheader("Detections")
                    for b in boxes:
                        st.write(
                            f"- {b['cls']} (id {b['cls_id']}): conf {b['conf']} — [{b['x1']},{b['y1']},{b['x2']},{b['y2']}]"
                        )
                else:
                    st.info("No objects detected.")

                # offer download button
                st.download_button(
                    "Download annotated image",
                    data=annotated_bytes,
                    file_name="annotated.png",
                    mime="image/png",
                )

# Optional: show API URL override (for remote deployments)
st.sidebar.header("Settings")
api_url_input = st.sidebar.text_input("API URL", value=API_URL)
if api_url_input != API_URL:
    st.sidebar.warning(
        "To change the API URL, update the `API_URL` in code or use Streamlit secrets."
    )
