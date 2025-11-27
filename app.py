import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse

app = FastAPI(title="Sticker Remover – Load Balancer Ready")

# Bind to PORT env var (RunPod default 8080)
port = int(os.getenv("PORT", 8000))
host = "0.0.0.0"  # Bind to all interfaces for proxy

# Load models at startup
print("Loading models...")
from remove_sticker_yolo_sam_fast import yolo, sam, predictor, lama_model
print("All models loaded!")

# Health check for RunPod (required for worker validation)
@app.get("/health")
def health():
    return {"status": "healthy", "gpu": "ready"}

@app.get("/ping")
def ping():
    return {"status": "ok", "gpu": "ready"}

@app.get("/")
def home():
    return {"message": "Sticker Remover API ready"}

@app.post("/process")
async def process_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})
    
    try:
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_out:
                contents = await file.read()
                tmp_in.write(contents)
                in_path = tmp_in.name
                out_path = tmp_out.name

        from remove_sticker_yolo_sam_fast import process_image
        process_image(in_path, out_path, slno=1)

        with open(out_path, "rb") as f:
            result = f.read()

        os.unlink(in_path)
        os.unlink(out_path)

        return Response(
            content=result,
            media_type="image/webp",
            headers={"Content-Disposition": f"inline; filename=cleaned{suffix}"}
        )

    except Exception as e:
        import traceback
        print("Error:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run Uvicorn on PORT (for RunPod)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=host, port=port)
