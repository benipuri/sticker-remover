# app.py — FINAL VERSION FOR RUNPOD LOAD BALANCER (2025)
import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse

app = FastAPI(title="Sticker Remover – RunPod Load Balancer")

# === CRITICAL: Bind to RunPod's expected port (8080) ===
# RunPod load balancer proxies traffic to $PORT (default 8080)
PORT = int(os.getenv("PORT", 8080))
HOST = "0.0.0.0"

print(f"Starting server on {HOST}:{PORT}...")

# === Load your heavy models once at startup (cold start) ===
print("Loading YOLO + SAM + LaMA models from remove_sticker_yolo_sam_fast.py...")
from remove_sticker_yolo_sam_fast import yolo, sam, predictor, lama_model
print("All models loaded successfully – endpoint ready for requests!")

# === Health checks (RunPod requires at least one of these) ===
@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def home():
    return {"message": "Sticker Remover API is running – send image to /process"}

# === Main processing endpoint ===
@app.post("/process")
async def process_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})

    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_out:
                contents = await file.read()
                tmp_in.write(contents)
                in_path = tmp_in.name
                out_path = tmp_out.name

        # Process with your model
        from remove_sticker_yolo_sam_fast import process_image
        process_image(in_path, out_path, slno=1)

        # Read and return cleaned image
        with open(out_path, "rb") as f:
            result = f.read()

        # Cleanup
        os.unlink(in_path)
        os.unlink(out_path)

        return Response(
            content=result,
            media_type="image/webp",
            headers={"Content-Disposition": f"attachment; filename=cleaned{suffix}"}
        )

    except Exception as e:
        import traceback
        print("ERROR in /process:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


# === Start Uvicorn on the correct port ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
