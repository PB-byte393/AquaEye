import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import os

# Import the Brain
from model.inference import AquaEyeBrain

# Initialize the App
app = FastAPI(title="AquaEye Nervous System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBAL BRAIN INSTANCE
# We define the path to the weights
WEIGHTS_PATH = os.path.join("weights", "aquaeye_final.pth")
brain = None

@app.on_event("startup")
async def startup_event():
    global brain
    # Check if weights exist
    if os.path.exists(WEIGHTS_PATH):
        try:
            brain = AquaEyeBrain(weights_path=WEIGHTS_PATH)
        except Exception as e:
            print(f"[FATAL] Model failed to initialize: {e}")
    else:
        print(f"[WARNING] Weights not found at {WEIGHTS_PATH}. Please upload 'aquaeye_final.pth' to backend/weights/.")

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    print(f"Received Analysis Request: {file.filename}")
    
    if brain is None:
        raise HTTPException(status_code=500, detail="AI Model not loaded.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image decoding failed")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid Image File")

    # Run Analysis
    try:
        # [UPDATED] Now returns heatmap too
        result_mask, heatmap, virtual_stain, particle_count, morphology = brain.analyze(img)
        
    except ValueError as e:
        error_msg = str(e)
        print(f"[REJECTION] {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    # ...

    # Encode Result Mask (Red Overlay)
    _, buffer_mask = cv2.imencode('.png', result_mask)
    mask_b64 = base64.b64encode(buffer_mask).decode('utf-8')
    
    # [NEW] Encode Heatmap (XAI)
    _, buffer_heat = cv2.imencode('.png', heatmap)
    heat_b64 = base64.b64encode(buffer_heat).decode('utf-8')

    # [NEW] Encode Virtual Stain
    _, buffer_stain = cv2.imencode('.png', virtual_stain)
    stain_b64 = base64.b64encode(buffer_stain).decode('utf-8')
    
    print(f"Analysis Complete. Detected {particle_count} particles.")

    return {
        "status": "success",
        "filename": file.filename,
        "particle_count": particle_count,
        "mask_url": f"data:image/png;base64,{mask_b64}",
        "heatmap_url": f"data:image/png;base64,{heat_b64}",
        "stain_url": f"data:image/png;base64,{stain_b64}", # <--- Sending to Frontend
        "distribution": {
            "labels": ["Fragments", "Fibers"], 
            "data": [morphology["fragment"], morphology["fiber"]]
        }
    }