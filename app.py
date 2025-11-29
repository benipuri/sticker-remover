import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse

app = FastAPI()

# --- DO NOT LOAD MODELS HERE ---
# We will load them lazily on first request only

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "Sticker Remover API ready"}

@app.post("/process")
async def process_endpoint(file: UploadFile = File(...)):
    # Load models only when first request comes in
    global process_image
    try:
        from remove_sam_lama_fast import process_image
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Model load failed: {str(e)}"})

    try:
        suffix = os.path.splitext(file.filename)[1] or ".webp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_out:
                contents = await file.read()
                tmp_in.write(contents)
                in_path = tmp_in.name
                out_path = tmp_out.name

        process_image(in_path, out_path, slno=1)

        with open(out_path, "rb") as f:
            result = f.read()

        os.unlink(in_path)
        os.unlink(out_path)

        return Response(content=result, media_type="image/webp")

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": traceback.format_exc()})

@app.post("/process-full")
async def process_full_endpoint(file: UploadFile = File(...)):
    # Load the full-inpaint variant lazily on first request
    try:
        from remove_sam_lama_fast import process_image_full
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model load failed (full): {str(e)}"},
        )

    try:
        suffix = os.path.splitext(file.filename)[1] or ".webp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_out:

            contents = await file.read()
            tmp_in.write(contents)
            in_path = tmp_in.name
            out_path = tmp_out.name

        # 🔥 Full-image inpaint
        process_image_full(in_path, out_path, slno=1)

        with open(out_path, "rb") as f:
            result = f.read()

        os.unlink(in_path)
        os.unlink(out_path)

        return Response(content=result, media_type="image/webp")

    except Exception:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": traceback.format_exc()},
        )

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 80))

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)
