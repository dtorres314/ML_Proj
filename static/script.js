let selectedFile = null;

// Load files
document.getElementById("loadFilesBtn").addEventListener("click", async () => {
  const response = await fetch("/load_files", { method: "POST" });
  const data = await response.json();
  displayResults("Available Files", data.files);
});

// Preprocess files
document.getElementById("preprocessBtn").addEventListener("click", async () => {
  const response = await fetch("/load_files", { method: "POST" });
  const data = await response.json();
  const files = data.files;

  if (files.length === 0) {
    alert("No XML files found for preprocessing.");
    return;
  }

  const responsePreprocess = await fetch("/preprocess", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files }),
  });

  const preprocessResults = await responsePreprocess.json();
  const resultsDiv = document.getElementById("results");

  if (preprocessResults.results && preprocessResults.results.length > 0) {
    const resultsHTML = preprocessResults.results
      .map(
        (result) =>
          `<div>
              <strong>File:</strong> ${result.file} <br />
              <strong>Status:</strong> ${result.status} <br />
              ${
                result.status === "Processed"
                  ? `<strong>Output Path:</strong> ${result.output}`
                  : `<strong>Error Details:</strong> ${result.details || "N/A"}`
              }
            </div><hr />`
      )
      .join("");

    resultsDiv.innerHTML = `<h2>Preprocessing Results:</h2>${resultsHTML}`;
  } else {
    resultsDiv.innerHTML =
      "<h2>Preprocessing Results:</h2><p>No results available.</p>";
  }
});

// Train model
document.getElementById("trainModelBtn").addEventListener("click", async () => {
  const response = await fetch("/train_model", { method: "POST" });
  const data = await response.json();
  displayResults("Training Results", data);
});

// Browse and upload XML file
document.getElementById("browseFileBtn").addEventListener("click", () => {
  document.getElementById("uploadFileInput").click();
});

document
  .getElementById("uploadFileInput")
  .addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (file) {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/upload_file", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      if (response.ok) {
        selectedFile = result.file;
        document.getElementById(
          "selectedFileName"
        ).textContent = `Selected File: ${file.name}`;
        document.getElementById("predictBtn").disabled = false;
      } else {
        alert(result.message);
      }
    }
  });

// Predict
document.getElementById("predictBtn").addEventListener("click", async () => {
  if (!selectedFile) return;

  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file: selectedFile }),
  });

  const prediction = await response.json();
  const resultsDiv = document.getElementById("results");

  if (response.ok) {
    resultsDiv.innerHTML = `
      <h2>Prediction Results:</h2>
      <p><strong>Problem ID:</strong> ${prediction.ProblemID}</p>
      <p><strong>Predicted Section ID:</strong> ${prediction.PredictedSectionID}</p>
      <p><strong>Book:</strong> ${prediction.Book}</p>
      <p><strong>Chapter:</strong> ${prediction.Chapter}</p>
      <p><strong>Section:</strong> ${prediction.Section}</p>
    `;
  } else {
    resultsDiv.innerHTML = `<h2>Error:</h2><pre>${prediction.details}</pre>`;
  }
});

function displayResults(title, data) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = `<h2>${title}:</h2><pre>${JSON.stringify(
    data,
    null,
    2
  )}</pre>`;
}
