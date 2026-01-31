import logging
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from pipeline import generate_mesh, remove_background

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="img2mesh")

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Track job status: job_id -> {"status": ..., "formats": [...], "error": ...}
jobs: dict[str, dict] = {}


def run_pipeline(job_id: str, image_bytes: bytes):
    """Run the full pipeline in a background thread."""
    try:
        jobs[job_id]["status"] = "removing_background"
        clean_image = remove_background(image_bytes)

        jobs[job_id]["status"] = "generating_mesh"
        output_dir = OUTPUTS_DIR / job_id
        results = generate_mesh(clean_image, output_dir, job_id)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["formats"] = list(results.keys())
    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/api/upload")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accept an image upload and start the 3D generation pipeline."""
    image_bytes = await file.read()
    job_id = uuid.uuid4().hex[:12]

    jobs[job_id] = {"status": "queued", "formats": [], "error": None}
    background_tasks.add_task(run_pipeline, job_id, image_bytes)

    return JSONResponse({"job_id": job_id})


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    """Poll the status of a generation job."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(jobs[job_id])


@app.get("/api/download/{job_id}.{fmt}")
async def download_model(job_id: str, fmt: str):
    """Download a generated model file."""
    if fmt not in ("glb", "obj", "stl"):
        return JSONResponse({"error": "Unsupported format"}, status_code=400)

    file_path = OUTPUTS_DIR / job_id / f"{job_id}.{fmt}"
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    media_types = {
        "glb": "model/gltf-binary",
        "obj": "text/plain",
        "stl": "application/octet-stream",
    }
    return FileResponse(
        path=str(file_path),
        filename=f"model.{fmt}",
        media_type=media_types.get(fmt, "application/octet-stream"),
    )


# Serve static frontend
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
