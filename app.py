import io
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import os
import uvicorn
from remove_sam_lama_fast import process_image   # your pipeline


app = FastAPI(title="Sticker Remover API")

# Allow all origins (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Health Endpoint (RunPod needs this!)
# -----------------------------
@app.get("/ping")
def ping():
    return {"status": "ok"}


# -----------------------------
# POST /process  (image upload)
# -----------------------------
@app.post("/process")
async def process_route(file: UploadFile = File(...)):
    """
    Accepts an uploaded image (jpg/png/webp).
    Returns base64 cleaned result.
    """
    img_bytes = await file.read()

    # Run your YOLO -> SAM -> LaMa pipeline
    output_bytes = process_image(img_bytes)

    # Convert to base64
    output_base64 = base64.b64encode(output_bytes).decode("utf-8")

    return JSONResponse({
        "status": "success",
        "result_base64": output_base64
    })


# -----------------------------
# PORT-AWARE STARTER (for RunPod & Modal)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
