import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse

app = FastAPI(title="Sticker Remover API – Serverless Ready")


# CRITICAL: Load all heavy models once at container startup
# This runs during cold start (5–8 seconds) and never again
print("Starting model loading (YOLO + SAM + LaMA)...")
from remove_sam_lama_fast import yolo, sam, predictor, lama_model  # ← triggers loading
print("All models successfully loaded into GPU memory!")


@app.get("/")
def home():
    return {"message": "Sticker Remover API is running (serverless mode)"}


@app.get("/ping")
def ping():
    return {"status": "ok", "gpu": "ready"}


@app.post("/process")
async def process_endpoint(file: UploadFile = File(...)):
    # Accept any image type (jpg, png, webp, etc.)
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})

    try:
        # Create temp files in /tmp (fast in-memory storage, auto-cleaned)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_in:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_out:
                # Save uploaded file
                contents = await file.read()
                tmp_in.write(contents)
                in_path = tmp_in.name
                out_path = tmp_out.name

        # Process the image
        from remove_sam_lama_fast import process_image
        process_image(in_path, out_path, slno=1)

        # Return cleaned image
        with open(out_path, "rb") as f:
            result_bytes = f.read()

        # Clean up temp files
        os.unlink(in_path)
        os.unlink(out_path)

        return Response(
            content=result_bytes,
            media_type="image/webp",
            headers={"Content-Disposition": f"inline; filename=cleaned_{file.filename}"}
        )

    except Exception as e:
        import traceback
        print("Error:", traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
