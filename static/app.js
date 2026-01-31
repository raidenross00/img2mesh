const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const statusEl = document.getElementById("status");
const viewerWrapper = document.getElementById("viewerWrapper");
const modelViewer = document.getElementById("modelViewer");
const downloads = document.getElementById("downloads");

// Drag-and-drop handlers
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) handleFile(file);
});

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        setStatus("Please upload an image file.", true);
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.style.display = "inline-block";
    };
    reader.readAsDataURL(file);

    // Reset UI
    viewerWrapper.style.display = "none";
    downloads.style.display = "none";

    uploadFile(file);
}

async function uploadFile(file) {
    setStatus('<span class="spinner"></span> Uploading...');

    const formData = new FormData();
    formData.append("file", file);

    let res;
    try {
        res = await fetch("/api/upload", { method: "POST", body: formData });
    } catch (err) {
        setStatus("Upload failed: " + err.message, true);
        return;
    }

    if (!res.ok) {
        setStatus("Upload failed: " + res.statusText, true);
        return;
    }

    const data = await res.json();
    pollStatus(data.job_id);
}

async function pollStatus(jobId) {
    const statusMessages = {
        queued: "Queued...",
        removing_background: "Removing background...",
        generating_mesh: "Generating 3D model (this may take a minute)...",
        done: "Done!",
    };

    const poll = async () => {
        let res;
        try {
            res = await fetch(`/api/status/${jobId}`);
        } catch {
            setStatus("Lost connection to server.", true);
            return;
        }

        const data = await res.json();

        if (data.status === "error") {
            setStatus("Error: " + (data.error || "Unknown error"), true);
            return;
        }

        if (data.status === "done") {
            setStatus("Model ready!");
            showResult(jobId, data.formats);
            return;
        }

        const msg = statusMessages[data.status] || data.status;
        setStatus('<span class="spinner"></span> ' + msg);
        setTimeout(poll, 1500);
    };

    poll();
}

function showResult(jobId, formats) {
    // Show 3D viewer with GLB
    const glbUrl = `/api/download/${jobId}.glb`;
    modelViewer.setAttribute("src", glbUrl);
    viewerWrapper.style.display = "block";

    // Show download links
    downloads.innerHTML = "";
    for (const fmt of formats) {
        const a = document.createElement("a");
        a.href = `/api/download/${jobId}.${fmt}`;
        a.download = `model.${fmt}`;
        a.textContent = `Download .${fmt.toUpperCase()}`;
        downloads.appendChild(a);
    }
    downloads.style.display = "flex";
}

function setStatus(html, isError = false) {
    statusEl.innerHTML = html;
    statusEl.className = "status" + (isError ? " error" : "");
}
