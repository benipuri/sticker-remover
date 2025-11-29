import os
import tempfile
import threading
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse

app = FastAPI()


# ============================================================
# 🔥 BACKGROUND PRELOAD ON STARTUP — NO COLD STARTS EVER AGAIN
# ============================================================

def preload_models():
    try:
        print("🔥 Preloading models on startup...")
        import remove_sam_lama_fast as m

        # Touch objects to ensure initialization
        _ = m.yolo
        _ = m.sam
        _ = m.lama_model

        print("✅ Models preloaded successfully!")
    except Exception as e:
        print("❌ Model preload failed:", str(e))


@app.on_event("startup")
def startup_event():
    threading.Thread(target=preload_models, daemon=True).start()


# ============================================================
# Endpoints
# ============================================================

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/")
def home():
    return {"message": "Sticker Remover API ready"}


# ============================================================
# ROI VERSION (Variant 1)
# ============================================================
@app.post("/process")
async def process_endpoint(file: UploadFile = File(...)):
    # Lazy import (already preloaded)
    try:
        from remove_sam_lama_fast import process_image
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Model load failed: {str(e)}"},
        )

    try:
        suffix = os.path.splitext(file.filename)[1] or ".webp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_out:

            tmp_in.write(await file.read())
            in_path = tmp_in.name
            out_path = tmp_out.name

        # ROI processing
        process_image(in_path, out_path, slno=1)

        result = open(out_path, "rb").read()

        os.unlink(in_path)
        os.unlink(out_path)

        return Response(
            content=result,
            media_type="image/webp",
            headers={"X-Status": "success"}
        )

    except Exception:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Processing failed",
                "trace": traceback.format_exc()
            }
        )


# ============================================================
# FULL IMAGE VERSION (Variant 2)
# ============================================================
@app.post("/process-full")
async def process_full_endpoint(file: UploadFile = File(...)):
    # Lazy import (already preloaded)
    try:
        from remove_sam_lama_fast import process_image_full
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Model load failed (full): {str(e)}"},
        )

    try:
        suffix = os.path.splitext(file.filename)[1] or ".webp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_out:

            tmp_in.write(await file.read())
            in_path = tmp_in.name
            out_path = tmp_out.name

        # FULL IMAGE PROCESSING
        process_image_full(in_path, out_path, slno=1)

        result = open(out_path, "rb").read()

        os.unlink(in_path)
        os.unlink(out_path)

        return Response(
            content=result,
            media_type="image/webp",
            headers={"X-Status": "success"}
        )

    except Exception:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Full inpaint failed",
                "trace": traceback.format_exc()
            }
        )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 80))
    uvicorn.run(app, host="0.0.0.0", port=port)
