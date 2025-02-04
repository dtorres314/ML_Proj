const loadFilesBtn = document.getElementById("loadFilesBtn");
const preprocessBtn = document.getElementById("preprocessBtn");
const trainModelBtn = document.getElementById("trainModelBtn");
const browseFileBtn = document.getElementById("browseFileBtn");
const uploadFileInput = document.getElementById("uploadFileInput");
const predictBtn = document.getElementById("predictBtn");
const selectedFileName = document.getElementById("selectedFileName");
const resultsDiv = document.getElementById("results");

let selectedFile = null;

function display(title, data) {
  resultsDiv.innerHTML = `<h2>${title}:</h2><pre>${JSON.stringify(
    data,
    null,
    2
  )}</pre>`;
}

// 1) Load XML Files
loadFilesBtn.addEventListener("click", async () => {
  const resp = await fetch("/load_files", { method: "POST" });
  const data = await resp.json();
  display("Available XML Files", data.files || []);
});

// 2) Preprocess
preprocessBtn.addEventListener("click", async () => {
  // First load the file list
  const loadResp = await fetch("/load_files", { method: "POST" });
  const loadData = await loadResp.json();
  const files = loadData.files || [];

  if (!files.length) {
    display("Error", { message: "No files found in data/." });
    return;
  }

  // Call /preprocess
  const preprocessResp = await fetch("/preprocess", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files }),
  });
  const preprocessData = await preprocessResp.json();
  display("Preprocessing Results", preprocessData);
});

// 3) Train
trainModelBtn.addEventListener("click", async () => {
  const resp = await fetch("/train_model", { method: "POST" });
  const data = await resp.json();
  display("Training Results", data);
});

// 4) Browse & Upload Single File
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
      selectedFile = uploadData.file; // e.g. "myProblem.xml"
      selectedFileName.textContent = `Selected File: ${file.name}`;
      predictBtn.disabled = false;
    } else {
      alert(uploadData.message || "Upload failed");
    }
  }
});

// 5) Predict
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const resp = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file: selectedFile }),
  });
  const data = await resp.json();

  if (resp.ok) {
    resultsDiv.innerHTML = `
      <h2>Prediction Results:</h2>
      <p><strong>File:</strong> ${data.file}</p>
      <p><strong>Book ID:</strong> ${data.book_id}</p>
      <p><strong>Chapter ID:</strong> ${data.chapter_id}</p>
      <p><strong>Section ID:</strong> ${data.section_id}</p>
    `;
  } else {
    resultsDiv.innerHTML = `
      <h2>Prediction Error</h2>
      <pre>${JSON.stringify(data, null, 2)}</pre>
    `;
  }
});
