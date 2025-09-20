# app.py
"""
Lightweight FastAPI app that runs a YOLO model (ultralytics) and provides a
/predict endpoint which can: detect objects, return annotated image, and
optionally mask/remove selected objects using bounding-box or segmentation
masks with different fill modes (inpaint, blur, color).

This file is intentionally self-contained to avoid previous syntax issues.
"""
import os
import tempfile
import base64
from typing import List, Optional

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Try to import ultralytics YOLO. If not available, raise a clear error when used.
try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - runtime dependency
    YOLO = None

# App and model
app = FastAPI()
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
model = None
if YOLO is not None and os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)


# Response model
class BoxInfo(BaseModel):
    cls: str
    cls_id: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int


class PredictResponse(BaseModel):
    boxes: List[BoxInfo]
    annotated_image_base64: str
    result_image_base64: str
    mask_image_base64: Optional[str] = None


# Helper: read image bytes into cv2 BGR numpy array
def read_imagefile_into_cv2(b: bytes):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    mask_objects: bool = False,
    remove_objects: bool = False,
    classes_to_remove: Optional[str] = None,
    dilate: int = 0,
    inpaint_method: str = "telea",
    box_indices: Optional[str] = None,
    fill_mode: str = "inpaint",
    blur_ksize: int = 21,
    fill_color: Optional[str] = None,
    use_segmentation_masks: bool = False,
):
    if model is None:
        return JSONResponse(status_code=500, content={"detail": "YOLO model not loaded on server"})

    contents = await file.read()
    # write to a temp file for model.predict convenience
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        results = model.predict(
            source=tmp_path,
            imgsz=1280,
            conf=conf_threshold,
            iou=0.45,
            device=model.device,
            verbose=False,
        )
        r = results[0]

        original_img = read_imagefile_into_cv2(contents)
        if original_img is None:
            return JSONResponse(status_code=400, content={"detail": "Could not decode uploaded image"})
        h, w = original_img.shape[:2]

        boxes_out = []
        mask = np.zeros((h, w), dtype=np.uint8)

        # parse classes_to_remove
        classes_filter = None
        if classes_to_remove:
            classes_filter = set([c.strip() for c in classes_to_remove.split(",") if c.strip() != ""])

        # parse box indices
        indices_filter = None
        if box_indices:
            try:
                indices_filter = set(int(i.strip()) for i in box_indices.split(",") if i.strip() != "")
            except Exception:
                indices_filter = None

        # try to extract segmentation masks if requested
        masks_per_box = None
        if use_segmentation_masks and getattr(r, "masks", None) is not None:
            try:
                raw = getattr(r.masks, "masks", None) or getattr(r.masks, "data", None)
                if raw is not None:
                    masks_per_box = []
                    for m in raw:
                        try:
                            arr = m.numpy() if hasattr(m, "numpy") else np.array(m)
                        except Exception:
                            arr = np.asarray(m)
                        if arr.ndim == 2 and arr.shape[0] == h and arr.shape[1] == w:
                            masks_per_box.append((arr > 0).astype(np.uint8) * 255)
                        else:
                            masks_per_box = None
                            break
            except Exception:
                masks_per_box = None

        # create an annotated copy to draw boxes/labels on
        annotated_img = original_img.copy()

        for idx, box in enumerate(getattr(r, "boxes", [])):
            try:
                xyxy = box.xyxy[0].tolist()
            except Exception:
                continue
            x1, y1, x2, y2 = map(lambda v: int(round(v)), xyxy)
            cls_id = int(box.cls[0].item()) if hasattr(box, "cls") else -1
            conf = float(box.conf[0].item()) if hasattr(box, "conf") else 0.0
            cls_name = (model.names[cls_id] if model.names and cls_id in model.names else str(cls_id))

            boxes_out.append({"cls": cls_name, "cls_id": cls_id, "conf": round(conf, 4), "x1": x1, "y1": y1, "x2": x2, "y2": y2})

            should_mask = False
            if mask_objects or remove_objects:
                if classes_filter is None:
                    should_mask = True
                else:
                    if str(cls_id) in classes_filter or cls_name in classes_filter:
                        should_mask = True

            mask_this_box = should_mask
            if indices_filter is not None:
                mask_this_box = idx in indices_filter and should_mask

            if mask_this_box:
                if masks_per_box is not None and idx < len(masks_per_box):
                    mask = cv2.bitwise_or(mask, masks_per_box[idx])
                else:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            # draw on annotated image
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(annotated_img, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, annotated_buffer = cv2.imencode(".png", annotated_img)
        annotated_b64 = base64.b64encode(annotated_buffer.tobytes()).decode("utf-8")

        # apply removal on a fresh copy of the original image so annotations are not preserved
        result_img = original_img.copy()
        if remove_objects and np.any(mask > 0):
            if dilate and dilate > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
                mask = cv2.dilate(mask, kernel, iterations=1)

            if fill_mode == "inpaint":
                method = cv2.INPAINT_TELEA if inpaint_method.lower().startswith("t") else cv2.INPAINT_NS
                result_img = cv2.inpaint(result_img, mask, 3, method)
            elif fill_mode == "blur":
                k = blur_ksize if blur_ksize and blur_ksize % 2 == 1 else (blur_ksize + 1 if blur_ksize % 2 == 0 else 21)
                blurred = cv2.GaussianBlur(result_img, (k, k), 0)
                result_img[mask > 0] = blurred[mask > 0]
            elif fill_mode == "color":
                color_bgr = None
                if fill_color:
                    try:
                        if fill_color.startswith("#") and len(fill_color) == 7:
                            r = int(fill_color[1:3], 16)
                            g = int(fill_color[3:5], 16)
                            b = int(fill_color[5:7], 16)
                            color_bgr = (b, g, r)
                        else:
                            parts = [int(x.strip()) for x in fill_color.split(",")]
                            if len(parts) == 3:
                                color_bgr = (parts[2], parts[1], parts[0])
                    except Exception:
                        color_bgr = None

                if color_bgr is None:
                    color_bgr = (0, 0, 0)

                result_img[mask > 0] = color_bgr

        _, result_buffer = cv2.imencode(".png", result_img)
        result_b64 = base64.b64encode(result_buffer.tobytes()).decode("utf-8")

        response = {"boxes": boxes_out, "annotated_image_base64": annotated_b64, "result_image_base64": result_b64}
        if mask_objects:
            _, mask_buffer = cv2.imencode(".png", mask)
            mask_b64 = base64.b64encode(mask_buffer.tobytes()).decode("utf-8")
            response["mask_image_base64"] = mask_b64

        return JSONResponse(content=response)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
