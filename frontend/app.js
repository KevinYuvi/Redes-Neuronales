const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "white";
ctx.lineWidth = 16;
ctx.lineCap = "round";

let drawing = false;

function getPos(e) {
    const rect = canvas.getBoundingClientRect();

    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX;
    let clientY;

    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }

    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

function start(e) {
    drawing = true;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}

function draw(e) {
    if (!drawing) return;
    e.preventDefault();

    const pos = getPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
}

function stop() {
    drawing = false;
}

canvas.addEventListener("mousedown", start);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stop);
canvas.addEventListener("mouseleave", stop);

canvas.addEventListener("touchstart", start);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", stop);

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("results").innerHTML =
        `<p class="empty">Aún no hay predicción.</p>`;
}

async function sendImage() {
    const loading = document.getElementById("loading");
    const results = document.getElementById("results");

    loading.classList.remove("hidden");
    results.innerHTML = "";

    canvas.toBlob(async function (blob) {
        const formData = new FormData();
        formData.append("file", blob, "numero.png");

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            loading.classList.add("hidden");
            renderResults(data);

        } catch (error) {
            loading.classList.add("hidden");
            results.innerHTML = `
                <div class="result-box">
                    <strong>Error:</strong> no se pudo conectar con la API.
                </div>
            `;
        }
    });
}

function renderResults(data) {
    const results = document.getElementById("results");

results.innerHTML = `
    ${renderBox(
        "Python",
        data.python,
        "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg"
    )}
    ${renderBox(
        "Java",
        data.java,
        "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/java/java-original.svg"
    )}
    ${renderBox(
        "Rust",
        data.rust,
        "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/rust/rust-original.svg"
    )}
`;
}

function renderBox(name, result, logoPath) {
    if (result.error) {
        return `
            <div class="result-box">
                <div class="result-header">
                    <img class="logo-img" src="${logoPath}" alt="${name}">
                    <h3>${name}</h3>
                </div>
                <p>Error: ${result.error}</p>
            </div>
        `;
    }

    const confidence = result.confidence
        ? `<p class="confidence">Confianza: ${result.confidence}%</p>`
        : `<p class="confidence">Confianza no calculada</p>`;

    const time = result.time_ms
        ? `<p class="confidence">Tiempo: ${result.time_ms} ms</p>`
        : "";

    return `
        <div class="result-box">
            <div class="result-header">
                <img class="logo-img" src="${logoPath}" alt="${name}">
                <h3>${name}</h3>
            </div>

            <div class="prediction">${result.prediction}</div>
            ${confidence}
            ${time}
        </div>
    `;
}