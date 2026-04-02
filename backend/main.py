import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import cv2
import numpy as np
import uuid
import os
import tempfile
from model.inference import AquaEyeBrain
from model.reporting import AquaEyeReport

# --- GLOBAL VARIABLES ---
WEIGHTS_PATH = os.path.join("weights", "aquaeye_final.pth")
brain = None
RESULT_CACHE = {}

# [LIFESPAN MANAGER]
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    global brain
    print("[SYSTEM] Waking up AquaEye Neural Core...")
    if os.path.exists(WEIGHTS_PATH):
        try:
            brain = AquaEyeBrain(weights_path=WEIGHTS_PATH)
            print(f"[SUCCESS] Brain initialized from {WEIGHTS_PATH}")
        except Exception as e:
            print(f"[FATAL] Model failed to load: {e}")
    else:
        print(f"[WARNING] Weights file not found at {WEIGHTS_PATH}")
    
    yield # Server runs here
    
    # --- SHUTDOWN ---
    print("[SYSTEM] Shutting down. Clearing memory...")
    brain = None
    RESULT_CACHE.clear()

# Initialize App
app = FastAPI(title="AquaEye Nervous System", lifespan=lifespan)

# CORS SETUP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    if brain is None: 
        raise HTTPException(500, "AI Model not loaded.")

    # 1. Decode Image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("Decoding failed")
    except Exception:
        raise HTTPException(400, "Invalid Image File")

    # 2. Inference
    try:
        mask, heatmap, stain, stats = brain.analyze(img)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 3. Cache Results (RAM)
    request_id = str(uuid.uuid4())
    RESULT_CACHE[request_id] = {
        "raw": cv2.imencode(".png", img)[1].tobytes(),
        "mask": cv2.imencode(".png", mask)[1].tobytes(),
        "heatmap": cv2.imencode(".png", heatmap)[1].tobytes(),
        "stain": cv2.imencode(".png", stain)[1].tobytes(),
        "stats": stats
    }

    return {
        "status": "success",
        "request_id": request_id,
        "particle_count": stats["accepted_count"],
        "distribution": {
            "labels": ["Fragments", "Fibers"], 
            "data": [stats["morphology"]["fragment"], stats["morphology"]["fiber"]]
        }
    }

# [GOD-LEVEL FIX] DIRECT MEMORY STREAMING
# We use 'Response' (bytes) instead of 'FileResponse' (disk path)
# This serves images directly from RAM -> Network (Zero Latency)
@app.get("/results/{request_id}/{layer_type}")
async def get_result_layer(request_id: str, layer_type: str):
    if request_id not in RESULT_CACHE:
        raise HTTPException(404, "Session Expired")
    
    if layer_type not in RESULT_CACHE[request_id]:
        raise HTTPException(404, "Layer not found")
    
    # SERVE DIRECTLY FROM RAM
    return Response(
        content=RESULT_CACHE[request_id][layer_type], 
        media_type="image/png"
    )

@app.get("/report/{request_id}")
async def generate_report_endpoint(request_id: str):
    if request_id not in RESULT_CACHE:
        raise HTTPException(404, "Report data not found (Session Expired)")
    
    data = RESULT_CACHE[request_id]
    
    # Create temp files ONLY for the PDF generator (FPDF needs disk paths)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_raw:
        f_raw.write(data["raw"])
        raw_path = f_raw.name
        
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_mask:
        f_mask.write(data["mask"])
        mask_path = f_mask.name

    # Generate PDF
    pdf_path = f"report_{request_id}.pdf"
    report = AquaEyeReport(request_id, data["stats"])
    report.generate(raw_path, mask_path, pdf_path)
    
    # Clean up temp images? Optional. For now we leave them or OS cleans them.
    
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"CoA_{request_id}.pdf")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)