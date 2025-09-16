# app.py
import os
import base64
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List

# Configure
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "models/best.pt"
)  # set env var or default path

app = FastAPI(title="Furniture Detection API")

# Load model once
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")


class Box(BaseModel):
    cls: str
    cls_id: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int


class PredictResponse(BaseModel):
    boxes: List[Box]
    annotated_image_base64: str
    mask_image_base64: str = None


def read_imagefile_into_cv2(file_bytes: bytes):
    # read bytes and convert to cv2 image (BGR)
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    mask_objects: bool = False,
    remove_objects: bool = False,
):
    # Read file bytes
    contents = await file.read()
    # Save to temp file because ultralytics predict works well with paths (also can accept arrays)
    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(file.filename)[1], delete=False
    ) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Perform inference (single image)
        results = model.predict(
            source=tmp_path,
            imgsz=1280,
            conf=conf_threshold,
            iou=0.45,
            device=model.device,
            verbose=False,
        )
        r = results[0]

        # load original bytes into cv2 for drawing
        img = read_imagefile_into_cv2(contents)
        h, w = img.shape[:2]

        boxes_out = []
        mask = np.zeros((h, w), dtype=np.uint8)  # mask for objects

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                # xyxy tensor: (x1,y1,x2,y2)
                xyxy = box.xyxy[0].tolist()  # list of 4 floats
                x1, y1, x2, y2 = map(lambda v: int(round(v)), xyxy)
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = (
                    model.names[cls_id]
                    if model.names and cls_id in model.names
                    else str(cls_id)
                )

                boxes_out.append(
                    {
                        "cls": cls_name,
                        "cls_id": cls_id,
                        "conf": round(conf, 4),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

                if mask_objects or remove_objects:
                    # Create mask for this object (rectangle)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # white rectangle

                # Draw box + label on annotated image
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    label,
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # If remove_objects, inpaint the masked areas
        if remove_objects and np.any(mask > 0):
            img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        # encode annotated image as base64 PNG
        _, buffer = cv2.imencode(".png", img)
        annotated_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        # If mask_objects, encode mask as base64
        mask_b64 = None
        if mask_objects:
            _, mask_buffer = cv2.imencode(".png", mask)
            mask_b64 = base64.b64encode(mask_buffer.tobytes()).decode("utf-8")

        response = {"boxes": boxes_out, "annotated_image_base64": annotated_b64}
        if mask_b64:
            response["mask_image_base64"] = mask_b64

        return JSONResponse(content=response)

    finally:
        # cleanup temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# Health
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
