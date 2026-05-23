# Redes Neuronales

Sistema de reconocimiento de dígitos escritos a mano utilizando redes neuronales convolucionales (CNN) implementadas en:

- Python + PyTorch
- Rust + Candle

El sistema permite dibujar números desde navegador o celular y comparar las predicciones realizadas por ambos lenguajes.

---

# Tecnologías utilizadas

## Backend

- FastAPI
- PyTorch
- Pillow
- Rust Candle

## Frontend

- HTML
- CSS
- JavaScript Canvas API

## IA

- CNN (Convolutional Neural Network)
- Dataset MNIST

---

# Estructura del proyecto

```text
Redes Neuronales/
│
├── backend/
│   ├── __init__.py
│   ├── api.py
│   ├── model.py
│   └── mnist_model.pth
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── rust/
│   ├── Cargo.toml
│   ├── mnist_model.safetensors
│   ├── target/
│   │   └── release/
│   │       └── predict_image.exe
│   │
│   └── src/
│       └── bin/
│           ├── train.rs
│           ├── predict.rs
│           └── predict_image.rs
│
├── requirements.txt
└── MANUAL_INSTALACION.md
```

---

# Requisitos previos

Instalar previamente:

- Python 3.11+
- Rust
- Cargo

Verificar instalaciones:

```bash
python --version
rustc --version
cargo --version
```

---

# requirements.txt

```txt
fastapi
uvicorn
pillow
python-multipart
torch
torchvision
```

---

# Crear entorno virtual

Desde la carpeta principal:

```bash
python -m venv venv
```

Activar entorno virtual en Windows:

```bash
venv\Scripts\activate
```

---

# Instalar dependencias Python

```bash
pip install -r requirements.txt
```

---

# Instalar dependencias Rust

Entrar a carpeta Rust:

```bash
cd rust
```

Compilar proyecto:

```bash
cargo build --release --bin predict_image
```

Esto generará:

```text
rust/target/release/predict_image.exe
```

Regresar a la carpeta principal:

```bash
cd ..
```

---

# Ejecutar el backend

Desde la carpeta principal:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Abrir en navegador:

```text
http://127.0.0.1:8000
```

---

# Ejecutar desde celular en la misma red WiFi

Obtener IP local:

```bash
ipconfig
```

Buscar IPv4:

```text
192.168.1.20
```

Abrir desde celular:

```text
http://192.168.1.20:8000
```

---

# Ejecutar desde internet usando ngrok

Descargar ngrok:

```text
https://ngrok.com
```

Con el backend ejecutándose:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Abrir otra terminal:

```bash
ngrok http 8000
```

ngrok generará algo como:

```text
https://xxxx.ngrok-free.app
```

Abrir esa URL desde el celular.

---

# Funcionamiento del sistema

1. El usuario dibuja un número del 0 al 9.
2. El frontend captura la imagen.
3. La imagen se envía al backend mediante FastAPI.
4. Python y Rust procesan la imagen.
5. Ambos modelos devuelven una predicción.
6. Los resultados se muestran visualmente en pantalla.

---

# Arquitectura del sistema

```text
Celular / Navegador
        ↓
Frontend HTML/CSS/JS
        ↓
FastAPI
        ↓
Python PyTorch / Rust Candle
        ↓
Predicción IA
```

---

# Modelo de IA utilizado

Se utilizó una red neuronal convolucional (CNN).

La arquitectura contiene:

- Capas convolucionales
- ReLU
- MaxPooling
- Capas fully connected

El modelo fue entrenado con:

- Dataset MNIST
- 60.000 imágenes de entrenamiento
- 10.000 imágenes de prueba

---

# Resultados obtenidos

## Python

- Accuracy: 98.58%
- Entrenamiento: 378 segundos
- Inferencia: 3.8 segundos

## Rust

- Accuracy: 94.45%
- Entrenamiento: 93 segundos
- Inferencia: 6.8 segundos

---
