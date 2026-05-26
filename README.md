# Redes Neuronales - Comparativa Python, Java y Rust

Sistema de reconocimiento de dígitos escritos a mano mediante redes neuronales convolucionales (CNN), implementado en tres lenguajes distintos:

- Python + PyTorch + CUDA
- Java + Deeplearning4j + Spring Boot
- Rust + Candle

El sistema permite dibujar números desde navegador o celular y comparar en tiempo real:

- Predicción
- Nivel de confianza
- Tiempo de inferencia

---

# Objetivo

Comparar el rendimiento de distintos lenguajes y frameworks de inteligencia artificial utilizando el dataset MNIST, evaluando:

- Precisión (Accuracy)
- Tiempo de entrenamiento
- Tiempo de inferencia
- Integración en aplicaciones web

---

# Tecnologías utilizadas

## Backend

- FastAPI
- Spring Boot
- Requests
- Pillow

## Inteligencia Artificial

### Python

- PyTorch
- CUDA 12.8
- NVIDIA RTX 3050 Laptop GPU

### Java

- Deeplearning4j (DL4J)
- ND4J
- Spring Boot

### Rust

- Candle

## Frontend

- HTML5
- CSS3
- JavaScript
- Canvas API

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
├── java_service/
│   ├── pom.xml
│   ├── mnist_model_java.zip
│   │
│   └── src/
│       └── main/
│           └── java/
│               └── com/
│                   └── demo/
│                       ├── JavaMnistApplication.java
│                       ├── MnistService.java
│                       └── PredictionController.java
│
├── rust/
│   ├── Cargo.toml
│   ├── mnist_model.safetensors
│   │
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
├── README.md
└── MANUAL_INSTALACION.md
```

---

# Requisitos previos

Instalar:

- Python 3.11+
- Java JDK 17+
- Maven 3.9+
- Rust
- Cargo
- NVIDIA Driver (opcional para CUDA)

Verificar instalación:

```bash
python --version
java -version
mvn -version
rustc --version
cargo --version
```

---

# Crear entorno virtual

```bash
python -m venv venv
```

Activar:

```bash
venv\Scripts\activate
```

---

# requirements.txt

```txt
fastapi
uvicorn
pillow
python-multipart
requests
torch
torchvision
```

---

# Instalar dependencias Python

```bash
pip install -r requirements.txt
```

---

# Configurar PyTorch con GPU NVIDIA

Verificar GPU:

```bash
nvidia-smi
```

Instalar versión CUDA:

```bash
pip uninstall torch torchvision torchaudio

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verificar:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Resultado esperado:

```text
2.x.x+cu128
12.8
True
NVIDIA GeForce RTX 3050 Laptop GPU
```

---

# Instalar y ejecutar Java

Entrar al proyecto:

```bash
cd java_service
```

Descargar dependencias:

```bash
mvn clean install
```

Ejecutar servicio:

```bash
mvn spring-boot:run
```

Resultado esperado:

```text
Modelo Java cargado en memoria
Tomcat started on port 8081
```

---

# Instalar Rust

Entrar:

```bash
cd rust
```

Compilar:

```bash
cargo build --release --bin predict_image
```

Se generará:

```text
rust/target/release/predict_image.exe
```

---

# Entrenar modelos

## Python

```bash
cd python

python main.py
```

Genera:

```text
mnist_model.pth
```

---

## Java

```bash
cd java_service

mvn exec:java "-Dexec.mainClass=Main"
```

Genera:

```text
mnist_model_java.zip
```

---

## Rust

```bash
cd rust

cargo run --release --bin train
```

Genera:

```text
mnist_model.safetensors
```

---

# Ejecutar sistema completo

## Terminal 1 - Java

```bash
cd java_service

mvn spring-boot:run
```

---

## Terminal 2 - FastAPI

Desde la raíz:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Abrir:

```text
http://127.0.0.1:8000
```

---

# Acceso desde celular

Obtener IP local:

```bash
ipconfig
```

Buscar IPv4:

```text
192.168.1.X
```

Abrir desde celular:

```text
http://192.168.1.X:8000
```

---

# Acceso desde Internet

Instalar ngrok.

Con FastAPI ejecutándose:

```bash
ngrok http 8000
```

Obtendrás una URL similar a:

```text
https://xxxx.ngrok-free.app
```

Abrir desde cualquier dispositivo.

---

# Arquitectura del sistema

```text
Celular / Navegador
        │
        ▼
Frontend HTML/CSS/JS
        │
        ▼
FastAPI
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
Python  Java   Rust
PyTorch DL4J  Candle
        │
        ▼
Resultado comparativo
```

---

# Funcionamiento

1. El usuario dibuja un número.
2. El frontend captura la imagen.
3. La imagen se envía a FastAPI.
4. Python procesa la imagen con PyTorch.
5. Java procesa la imagen mediante Spring Boot y Deeplearning4j.
6. Rust procesa la imagen mediante Candle.
7. Cada modelo devuelve:
   - Número predicho.
   - Confianza.
   - Tiempo de inferencia.
8. Los resultados se muestran en pantalla.

---

# Modelo utilizado

Se utilizó una CNN (Convolutional Neural Network).

Arquitectura:

- Capas convolucionales
- ReLU
- MaxPooling
- Capas Fully Connected
- Softmax

Dataset:

- MNIST
- 60.000 imágenes de entrenamiento
- 10.000 imágenes de prueba
- Imágenes de 28x28 píxeles

---

# Resultados obtenidos

## Entrenamiento

| Lenguaje | Framework | Accuracy | Tiempo |
|-----------|------------|-----------|---------|
| Python | PyTorch + CUDA | 98.78% | 68.03 s |
| Java | Deeplearning4j | 98.03% | 60.46 s |
| Rust | Candle | 94.45% | 93.51 s |

---

## Evaluación del conjunto de prueba

| Lenguaje | Tiempo |
|-----------|---------|
| Python | 2.2790 s |
| Java | 2.2820 s |
| Rust | 6.8601 s |

---

## Inferencia en producción

Modelo cargado en memoria:

| Lenguaje | Tiempo |
|-----------|---------|
| Java Spring Boot | 12.12 ms |
| Python CPU | 12.83 ms |
| Python GPU | 8.83 ms |
| Rust | 57.49 ms |

