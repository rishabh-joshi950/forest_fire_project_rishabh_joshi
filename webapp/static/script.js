// ================= Slider live labels =================
const sliderUnits = {
  temperature: "°C",
  humidity: "%",
  smoke_level: "",
  wind_speed: " km/h",
  rainfall: " mm",
};

Object.keys(sliderUnits).forEach((id) => {
  const input = document.getElementById(id);
  const out = document.getElementById(id + "-out");
  const update = () => (out.textContent = input.value + sliderUnits[id]);
  input.addEventListener("input", update);
  update();
});

// ================= Gauge tick marks (drawn once) =================
const ticksGroup = document.querySelector(".gauge__ticks");
const cx = 120, cy = 120, r1 = 100, r2 = 92;
for (let i = 0; i <= 10; i++) {
  const angle = Math.PI - (i / 10) * Math.PI; // 180deg -> 0deg
  const x1 = cx + r1 * Math.cos(angle);
  const y1 = cy - r1 * Math.sin(angle);
  const x2 = cx + r2 * Math.cos(angle);
  const y2 = cy - r2 * Math.sin(angle);
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke", "#5a6656");
  line.setAttribute("stroke-width", "1.5");
  ticksGroup.appendChild(line);
}

// ================= Needle rotation =================
// Needle rests pointing straight up (-90deg from horizontal) at probability 0.
// It sweeps from -90deg (left, safe) to +90deg (right, alert) as probability -> 1.
function setNeedle(prob) {
  const angle = -90 + prob * 180;
  document.getElementById("needleGroup").style.transform = `rotate(${angle}deg)`;
}
setNeedle(0); // start at rest

// ================= Sensor form submit =================
const form = document.getElementById("sensorForm");
const probValue = document.getElementById("probValue");
const riskLabel = document.getElementById("riskLabel");
const sensorHint = document.getElementById("sensorHint");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  sensorHint.textContent = "Reading sensors…";

  const payload = {
    temperature: document.getElementById("temperature").value,
    humidity: document.getElementById("humidity").value,
    smoke_level: document.getElementById("smoke_level").value,
    wind_speed: document.getElementById("wind_speed").value,
    rainfall: document.getElementById("rainfall").value,
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.error) {
      sensorHint.textContent = data.error;
      return;
    }

    const pct = Math.round(data.probability * 100);
    probValue.textContent = pct + "%";
    riskLabel.textContent = data.risk_level + " RISK";
    riskLabel.style.color =
      data.risk_level === "HIGH" ? "#ff6a3d" :
      data.risk_level === "MODERATE" ? "#d9a441" : "#5fae72";

    setNeedle(data.probability);
    sensorHint.textContent = "Adjust the readings, then take a new reading.";
  } catch (err) {
    sensorHint.textContent = "Could not reach the server. Is app.py running?";
  }
});

// ================= Image upload =================
const dropzone = document.getElementById("dropzone");
const imageInput = document.getElementById("imageInput");
const imageResult = document.getElementById("imageResult");

dropzone.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", async () => {
  const file = imageInput.files[0];
  if (!file) return;

  const previewUrl = URL.createObjectURL(file);
  imageResult.innerHTML = `
    <img src="${previewUrl}" alt="Uploaded photo preview">
    <div class="image-result__text">Checking photo…</div>
  `;
  imageResult.classList.add("is-visible");

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch("/detect-image", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      imageResult.querySelector(".image-result__text").textContent = data.error;
      return;
    }

    const pct = Math.round(data.probability * 100);
    const tagClass = data.fire_detected ? "tag--alert" : "tag--safe";
    const tagText = data.fire_detected ? "FIRE DETECTED" : "NO FIRE DETECTED";
    const methodNote = data.method === "color_fallback"
      ? "<br><span style='color:#8a9682; font-size:11px;'>(color-based check — train the CNN for a deep-learning result)</span>"
      : "";

    imageResult.querySelector(".image-result__text").innerHTML = `
      <span class="tag ${tagClass}">${tagText}</span><br>
      Confidence: ${pct}%${methodNote}
    `;
  } catch (err) {
    imageResult.querySelector(".image-result__text").textContent =
      "Could not reach the server.";
  }
});
