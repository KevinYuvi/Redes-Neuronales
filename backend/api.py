import io
import json
import subprocess
from pathlib import Path
import time
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.model import predict_python

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
RUST_BIN = BASE_DIR / "rust" / "target" / "release" / "predict_image.exe"
JAVA_DIR = BASE_DIR / "java_service"

app = FastAPI(title="IA MNIST - Python vs Rust")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_java(image_path: Path):
    try:
        with open(image_path, "rb") as file:
            response = requests.post(
                "http://127.0.0.1:8081/predict",
                files={"file": file},
                timeout=10
            )

        return response.json()

    except Exception as e:
        return {
            "language": "Java",
            "error": str(e)
        }

def predict_rust(image_path: Path):
    if not RUST_BIN.exists():
        return {
            "language": "Rust",
            "error": "No existe predict_image.exe"
        }

    start = time.perf_counter()

    process = subprocess.run(
        [str(RUST_BIN), str(image_path)],
        cwd=BASE_DIR / "rust",
        capture_output=True,
        text=True
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    if process.returncode != 0:
        return {
            "language": "Rust",
            "error": process.stderr,
            "time_ms": elapsed_ms
        }

    data = json.loads(process.stdout.strip())

    return {
        "language": "Rust",
        "prediction": data.get("prediction"),
        "confidence": data.get("confidence", None),
        "time_ms": elapsed_ms
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    temp_path = BASE_DIR / "temp_number.png"
    java_temp_path = JAVA_DIR / "temp_number.png"

    image.save(temp_path)
    image.save(java_temp_path)

    start = time.perf_counter()
    python_result = predict_python(image)
    python_result["time_ms"] = round((time.perf_counter() - start) * 1000, 2)
    
    rust_result = predict_rust(temp_path)
    java_result = predict_java(java_temp_path)

    return {
        "python": python_result,
        "java": java_result,
        "rust": rust_result
    }


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")