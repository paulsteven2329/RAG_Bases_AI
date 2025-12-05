const API_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const ingestBtn = document.getElementById("ingestBtn");
  const spinner = document.getElementById("spinner");
  const uploadStatus = document.getElementById("uploadStatus");
  const questionInput = document.getElementById("questionInput");
  const sendBtn = document.getElementById("sendBtn");
  const chatMessages = document.getElementById("chatMessages");

  let selectedFiles = [];

  // === UPLOAD ===
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    handleFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener("change", () => handleFiles(fileInput.files));

  function handleFiles(files) {
   // In handleFiles()
selectedFiles = Array.from(files).filter(f => 
  ['.txt','.pdf','.png','.jpg','.jpeg'].some(ext => f.name.toLowerCase().endsWith(ext))
);
    if (selectedFiles.length > 0) {
      dropZone.innerHTML = `<i class="fas fa-check-circle"></i><p>${selectedFiles.length} file(s) selected</p>`;
      ingestBtn.disabled = false;
    }
  }

  ingestBtn.addEventListener("click", async () => {
    if (selectedFiles.length === 0) return;

    spinner.classList.add("active");
    ingestBtn.disabled = true;
    uploadStatus.textContent = "Uploading and ingesting...";

    const formData = new FormData();
    selectedFiles.forEach(file => formData.append("file", file));

    try {
      const res = await fetch(`${API_URL}/ingest`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      uploadStatus.textContent = `Success: ${data.file} — ingestion started!`;
      uploadStatus.style.color = "green";
      addMessage("System", `Document "${data.file}" uploaded and being processed.`);
    } catch (err) {
      uploadStatus.textContent = "Upload failed.";
      uploadStatus.style.color = "red";
    } finally {
      spinner.classList.remove("active");
      ingestBtn.disabled = false;
      selectedFiles = [];
      dropZone.innerHTML = `<i class="fas fa-cloud-upload-alt"></i><p>Drop files here or <span class="link">click to upload</span></p>`;
    }
  });

  // === CHAT ===
  sendBtn.addEventListener("click", sendQuestion);
  questionInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendQuestion();
  });

  async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    addMessage("You", question);
    questionInput.value = "";
    sendBtn.disabled = true;

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      addMessage("Assistant", data.answer, data.sources);
    } catch (err) {
      addMessage("Assistant", "Sorry, something went wrong.");
    } finally {
      sendBtn.disabled = false;
      questionInput.focus();
    }
  }

  function addMessage(sender, text, sources = []) {
    const msg = document.createElement("div");
    msg.className = `message ${sender === "You" ? "user" : "bot"}`;

    const content = document.createElement("div");
    content.innerHTML = text.replace(/\n/g, "<br>");

    msg.appendChild(content);

    if (sources.length > 0) {
      const srcDiv = document.createElement("div");
      srcDiv.className = "sources";
      srcDiv.innerHTML = "Sources: " + sources.map(s =>
        `<a href="#" title="${s.source}">[chunk ${s.chunk}]</a>`
      ).join(", ");
      msg.appendChild(srcDiv);
    }

    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
});