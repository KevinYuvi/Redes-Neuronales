import io
import json
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.model import predict_python

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
RUST_BIN = BASE_DIR / "rust" / "target" / "release" / "predict_image.exe"

app = FastAPI(title="IA MNIST - Python vs Rust")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict_rust(image_path: Path):
    if not RUST_BIN.exists():
        return {
            "language": "Rust",
            "error": "No existe predict_image.exe"
        }

    process = subprocess.run(
        [str(RUST_BIN), str(image_path)],
        cwd=BASE_DIR / "rust",
        capture_output=True,
        text=True
    )

    if process.returncode != 0:
        return {
            "language": "Rust",
            "error": process.stderr
        }

    data = json.loads(process.stdout.strip())

    return {
        "language": "Rust",
        "prediction": data.get("prediction"),
        "confidence": data.get("confidence", None)
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    temp_path = BASE_DIR / "temp_number.png"
    image.save(temp_path)

    python_result = predict_python(image)
    rust_result = predict_rust(temp_path)

    return {
        "python": python_result,
        "rust": rust_result
    }


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")