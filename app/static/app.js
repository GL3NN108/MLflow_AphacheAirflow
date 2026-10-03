async function getJson(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || res.statusText);
  }
  return data;
}

function pill(ok, label) {
  return `<span class="${ok ? "ok" : "bad"}">${ok ? "connected" : "offline"}</span> · ${label}`;
}

function table(headers, rows) {
  if (!rows.length) return "Nothing to show yet.";
  const head = headers.map((h) => `<th>${h}</th>`).join("");
  const body = rows
    .map((row) => `<tr>${row.map((cell) => `<td>${cell ?? "—"}</td>`).join("")}</tr>`)
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function fmtTime(value) {
  if (!value) return "—";
  const date = typeof value === "number" ? new Date(value) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

async function refreshStatus() {
  const data = await getJson("/api/status");
  document.getElementById("mlflow-status").innerHTML = pill(data.mlflow.ok, data.mlflow.url);
  document.getElementById("airflow-status").innerHTML = pill(data.airflow.ok, data.airflow.url);
  document.getElementById("model-name").textContent = data.model_name;
  document.getElementById("mlflow-link").href = data.mlflow.url;
  document.getElementById("airflow-link").href = data.airflow.url;
}

async function refreshRuns() {
  const data = await getJson("/api/runs");
  document.getElementById("runs").innerHTML = table(
    ["Run", "Status", "Accuracy", "F1", "Started"],
    data.runs.map((run) => [
      run.run_id.slice(0, 8),
      run.status,
      run.accuracy?.toFixed?.(3) ?? run.accuracy,
      run.f1_macro?.toFixed?.(3) ?? run.f1_macro,
      fmtTime(run.start_time),
    ])
  );
}

async function refreshDagRuns() {
  const data = await getJson("/api/dag-runs");
  document.getElementById("dag-runs").innerHTML = table(
    ["Run", "State", "Started"],
    data.dag_runs.map((run) => [run.dag_run_id, run.state, fmtTime(run.start_date)])
  );
}

async function refreshAll() {
  await Promise.all([refreshStatus(), refreshRuns(), refreshDagRuns()]);
}

function bindTrain(buttonId, url, message) {
  const button = document.getElementById(buttonId);
  button.addEventListener("click", async () => {
    const note = document.getElementById("train-message");
    button.disabled = true;
    note.textContent = "Working…";
    try {
      const data = await getJson(url, { method: "POST" });
      note.textContent = message(data);
      await refreshAll();
    } catch (err) {
      note.textContent = err.message;
    } finally {
      button.disabled = false;
    }
  });
}

bindTrain("train-airflow", "/api/train/airflow", (data) => `DAG started: ${data.dag_run_id || "ok"}`);
bindTrain("train-direct", "/api/train/direct", (data) => `Logged run ${data.run_id} · accuracy ${data.accuracy.toFixed(3)}`);

document.getElementById("predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const body = Object.fromEntries([...form.entries()].map(([k, v]) => [k, Number(v)]));
  const note = document.getElementById("predict-result");
  note.textContent = "Scoring…";
  try {
    const data = await getJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    note.textContent = `Prediction: ${data.label}`;
  } catch (err) {
    note.textContent = err.message;
  }
});

refreshAll();
setInterval(refreshAll, 8000);
