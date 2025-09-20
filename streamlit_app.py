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
classes_to_remove = st.text_input(
    "Classes to remove (comma-separated names or ids)", value=""
)
dilate = st.number_input("Mask dilation (pixels)", min_value=0, max_value=200, value=0)
inpaint_method = st.selectbox("Inpaint method", options=["telea", "ns"], index=0)

# state for detections
if "detections" not in st.session_state:
    st.session_state["detections"] = []
if "annotated_bytes" not in st.session_state:
    st.session_state["annotated_bytes"] = None

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    # Detect button (unique key)
    if st.button("Detect", key="detect_btn"):
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
                "remove_objects": False,
                "classes_to_remove": classes_to_remove,
            }
            try:
                resp = requests.post(API_URL, files=files, params=params, timeout=60)
                resp.raise_for_status()
            except Exception as e:
                st.error(f"API request failed: {e}")
            else:
                data = resp.json()
                annotated_b64 = data.get("annotated_image_base64")
                if annotated_b64:
                    annotated_bytes = base64.b64decode(annotated_b64)
                    st.session_state["annotated_bytes"] = annotated_bytes
                    annotated_img = Image.open(BytesIO(annotated_bytes))
                    st.image(
                        annotated_img,
                        caption="Detections (before removal)",
                        use_column_width=True,
                    )

                mask_b64 = data.get("mask_image_base64")
                if mask_b64:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_img = Image.open(BytesIO(mask_bytes))
                    st.image(mask_img, caption="Object Masks", use_column_width=True)

                boxes = data.get("boxes", [])
                st.session_state["detections"] = boxes

    # If detections exist, allow user to select which to remove
    if st.session_state.get("detections"):
        st.subheader("Detections")
        boxes = st.session_state["detections"]
        # Build label list for multiselect
        labels = [
            f"[{i}] {b['cls']} (id {b['cls_id']}): conf {b['conf']} — [{b['x1']},{b['y1']},{b['x2']},{b['y2']}]"
            for i, b in enumerate(boxes)
        ]
        selected_labels = st.multiselect(
            "Select objects to remove", options=labels, key="remove_multiselect"
        )

        if st.button("Remove Selected Objects", key="remove_btn"):
            # compute indices from selected labels
            to_remove_indices = []
            for lbl in selected_labels:
                # label format is "[i] ..."; extract the numeric index
                try:
                    idx = int(lbl.split("]")[0].strip("[ "))
                    to_remove_indices.append(idx)
                except Exception:
                    pass

            if not to_remove_indices:
                st.info("No boxes selected for removal.")
            else:
                with st.spinner("Removing selected objects..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }
                    # compute indices from selected labels
                    to_remove_indices = []
                    for lbl in selected_labels:
                        # label format is "[i] ..."; extract the numeric index
                        try:
                            idx = int(lbl.split("]")[0].strip("[ "))
                            to_remove_indices.append(idx)
                        except Exception:
                            pass

                    params = {
                        "conf_threshold": conf,
                        "mask_objects": mask_objects,
                        "remove_objects": True,
                        "classes_to_remove": classes_to_remove,
                        "dilate": int(dilate),
                        "inpaint_method": inpaint_method,
                        "box_indices": ",".join([str(i) for i in to_remove_indices]),
                    }
                    try:
                        resp = requests.post(
                            API_URL, files=files, params=params, timeout=60
                        )
                        resp.raise_for_status()
                    except Exception as e:
                        st.error(f"API request failed: {e}")
                    else:
                            data = resp.json()
                            # Prefer `result_image_base64` (post-processed) when available.
                            result_b64 = data.get("result_image_base64") or data.get("annotated_image_base64")
                            if result_b64:
                                removed_bytes = base64.b64decode(result_b64)
                                removed_img = Image.open(BytesIO(removed_bytes))
                                st.image(
                                    removed_img,
                                    caption="After removal",
                                    use_column_width=True,
                                )
                                st.download_button(
                                    "Download removed-image",
                                    data=removed_bytes,
                                    file_name="removed.png",
                                    mime="image/png",
                                    key="download_removed",
                                )
