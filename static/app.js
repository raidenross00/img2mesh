import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
// MTLLoader removed — we load texture manually for reliability

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const statusEl = document.getElementById("status");
const viewerWrapper = document.getElementById("viewerWrapper");
const canvas = document.getElementById("viewerCanvas");
const downloads = document.getElementById("downloads");

let renderer, scene, camera, controls, currentModel;

function initViewer() {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.position.set(0, 0.5, 2.5);

    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 3;
    controls.target.set(0, 0, 0);

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);
    const dir1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dir1.position.set(2, 3, 2);
    scene.add(dir1);
    const dir2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dir2.position.set(-2, 1, -1);
    scene.add(dir2);

    resizeViewer();
    animate();
}

function resizeViewer() {
    const w = viewerWrapper.clientWidth;
    const h = viewerWrapper.clientHeight || 500;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function loadModel(jobId) {
    if (!renderer) initViewer();

    // Remove previous model
    if (currentModel) {
        scene.remove(currentModel);
        currentModel = null;
    }

    // Ensure layout is complete before sizing
    requestAnimationFrame(() => resizeViewer());

    const objUrl = `/api/download/${jobId}.obj`;
    const textureUrl = `/api/assets/${jobId}/${jobId}_texture.png`;

    // Load texture directly (bypasses MTLLoader path resolution issues)
    console.log("[img2mesh] Loading texture:", textureUrl);
    const textureLoader = new THREE.TextureLoader();
    textureLoader.load(textureUrl, (texture) => {
        console.log("[img2mesh] Texture loaded:", texture.image.width, "x", texture.image.height);
        texture.colorSpace = THREE.SRGBColorSpace;

        const material = new THREE.MeshStandardMaterial({
            map: texture,
            roughness: 0.8,
            metalness: 0.0,
        });

        console.log("[img2mesh] Loading OBJ:", objUrl);
        const objLoader = new OBJLoader();
        objLoader.load(objUrl, (obj) => {
            console.log("[img2mesh] OBJ loaded, children:", obj.children.length);

            // Apply texture material to all meshes
            obj.traverse((child) => {
                if (child.isMesh) {
                    child.material = material;
                }
            });

            // Center and fit the model
            const box = new THREE.Box3().setFromObject(obj);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            console.log("[img2mesh] Model bounds:", size.x.toFixed(3), size.y.toFixed(3), size.z.toFixed(3));
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 1.5 / maxDim;
            obj.scale.setScalar(scale);
            obj.position.sub(center.multiplyScalar(scale));

            scene.add(obj);
            currentModel = obj;

            // Reset camera
            camera.position.set(0, 0.5, 2.5);
            controls.target.set(0, 0, 0);
            controls.update();
            resizeViewer();
            console.log("[img2mesh] Model added to scene");
        },
        undefined,
        (err) => console.error("[img2mesh] OBJ load error:", err)
        );
    },
    undefined,
    (err) => console.error("[img2mesh] Texture load error:", err)
    );
}

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

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.style.display = "inline-block";
    };
    reader.readAsDataURL(file);

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
    viewerWrapper.style.display = "block";
    loadModel(jobId);

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

window.addEventListener("resize", () => {
    if (renderer) resizeViewer();
});
