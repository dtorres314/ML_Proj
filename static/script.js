const loadFilesBtn = document.getElementById("loadFilesBtn");
const preprocessBtn = document.getElementById("preprocessBtn");
const trainModelBtn = document.getElementById("trainModelBtn");
const browseFileBtn = document.getElementById("browseFileBtn");
const uploadFileInput = document.getElementById("uploadFileInput");
const predictBtn = document.getElementById("predictBtn");
const selectedFileName = document.getElementById("selectedFileName");
const resultsDiv = document.getElementById("results");

let selectedFile = null;

// Helper to display JSON results nicely
function displayResults(title, data) {
  resultsDiv.innerHTML = `<h2>${title}:</h2><pre>${JSON.stringify(
    data,
    null,
    2
  )}</pre>`;
}

// 1. Load XML Files
loadFilesBtn.addEventListener("click", async () => {
  const response = await fetch("/load_files", { method: "POST" });
  const data = await response.json();
  displayResults("Available XML Files", data.files || []);
});

// 2. Preprocess Files
preprocessBtn.addEventListener("click", async () => {
  const loadResp = await fetch("/load_files", { method: "POST" });
  const loadData = await loadResp.json();
  const files = loadData.files || [];
  if (!files.length) {
    displayResults("Preprocess Error", {
      message: "No XML files found in 'data'.",
    });
    return;
  }

  const preprocessResp = await fetch("/preprocess", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files }),
  });
  const preprocessData = await preprocessResp.json();
  displayResults("Preprocessing Results", preprocessData);
});

// 3. Train Model
trainModelBtn.addEventListener("click", async () => {
  const resp = await fetch("/train_model", { method: "POST" });
  const data = await resp.json();
  displayResults("Training Results", data);
});

// 4. Browse & Upload Single XML File
browseFileBtn.addEventListener("click", () => {
  uploadFileInput.click();
});

uploadFileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) {
    const formData = new FormData();
    formData.append("file", file);

    const uploadResp = await fetch("/upload_file", {
      method: "POST",
      body: formData,
    });
    const uploadData = await uploadResp.json();

    if (uploadResp.ok) {
      selectedFile = uploadData.file;
      selectedFileName.textContent = `Selected File: ${file.name}`;
      predictBtn.disabled = false;
    } else {
      alert(uploadData.message || "Upload failed.");
    }
  }
});

// 5. Predict (Single File)
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  const resp = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file: selectedFile }),
  });
  const data = await resp.json();

  if (resp.ok) {
    // Show predicted Book, Chapter, Section
    const html = `
      <h2>Prediction Results:</h2>
      <div>
        <strong>File:</strong> ${data.file} <br />
        <strong>Book ID:</strong> ${data.book_id} <br />
        <strong>Chapter ID:</strong> ${data.chapter_id} <br />
        <strong>Section ID:</strong> ${data.section_id}
      </div>
    `;
    resultsDiv.innerHTML = html;
  } else {
    resultsDiv.innerHTML = `
      <h2>Prediction Error</h2>
      <p>Error: ${data.error}</p>
      <pre>${JSON.stringify(data.details, null, 2)}</pre>
    `;
  }
});
