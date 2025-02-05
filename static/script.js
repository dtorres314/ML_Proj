const loadFilesBtn = document.getElementById("loadFilesBtn");
const extractDBBtn = document.getElementById("extractDBBtn");
const trainTestBtn = document.getElementById("trainTestBtn");
const browseFileBtn = document.getElementById("browseFileBtn");
const uploadFileInput = document.getElementById("uploadFileInput");
const predictBtn = document.getElementById("predictBtn");
const selectedFileName = document.getElementById("selectedFileName");
const resultsDiv = document.getElementById("results");

let selectedFile = null;

function display(title, data) {
  resultsDiv.innerHTML = `<h2>${title}</h2><pre>${JSON.stringify(
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

// 2) Extract & Save to DB
extractDBBtn.addEventListener("click", async () => {
  // 2.1) Get list from load_files
  const loadResp = await fetch("/load_files", { method: "POST" });
  const loadData = await loadResp.json();
  const files = loadData.files || [];

  if (!files.length) {
    display("Error", { message: "No .xml found in data." });
    return;
  }

  // 2.2) Send to /extract_and_save_db
  const resp = await fetch("/extract_and_save_db", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files }),
  });
  const resultData = await resp.json();
  display("Extract & Save to DB Results", resultData);
});

// 3) Train & Test
trainTestBtn.addEventListener("click", async () => {
  const resp = await fetch("/train_and_test", { method: "POST" });
  const data = await resp.json();
  display("Train & Test Summary", data);
});

// 4) Browse Single File
browseFileBtn.addEventListener("click", () => {
  uploadFileInput.click();
});

uploadFileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) {
    const formData = new FormData();
    formData.append("file", file);

    const upResp = await fetch("/upload_file", {
      method: "POST",
      body: formData,
    });
    const upData = await upResp.json();
    if (upResp.ok) {
      selectedFileName.textContent = `Selected File: ${file.name}`;
      predictBtn.disabled = false;
      selectedFile = upData.file;
    } else {
      alert(upData.message || "Upload failed");
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
    // Show IDs & names
    resultsDiv.innerHTML = `
      <h2>Prediction Results</h2>
      <p><strong>File:</strong> ${data.file}</p>
      <p><strong>Book ID:</strong> ${data.book_id}</p>
      <p><strong>Book Name:</strong> ${data.book_name}</p>
      <p><strong>Chapter ID:</strong> ${data.chapter_id}</p>
      <p><strong>Chapter Name:</strong> ${data.chapter_name}</p>
      <p><strong>Section ID:</strong> ${data.section_id}</p>
      <p><strong>Section Name:</strong> ${data.section_name}</p>
    `;
  } else {
    // Show errors
    resultsDiv.innerHTML = `
      <h2>Prediction Error</h2>
      <pre>${JSON.stringify(data, null, 2)}</pre>
    `;
  }
});
