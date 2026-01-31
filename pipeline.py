import io
import logging
import os
import sys
import tempfile
from pathlib import Path

# Reduce VRAM fragmentation on GPUs with limited memory
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Add TripoSR to path
sys.path.insert(0, str(Path(__file__).parent / "TripoSR"))

import numpy as np
import torch
import trimesh
from PIL import Image
from rembg import remove

logger = logging.getLogger(__name__)

# Global model reference — loaded once on first use
_tsr_model = None


def get_tsr_model():
    """Lazy-load the TripoSR model."""
    global _tsr_model
    if _tsr_model is None:
        logger.info("Loading TripoSR model (first request, may take a moment)...")
        from tsr.system import TSR

        _tsr_model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tsr_model.to(device)
        # Enable renderer chunking to limit VRAM usage during mesh extraction
        _tsr_model.renderer.set_chunk_size(8192)
        logger.info(f"TripoSR model loaded on {device}")
    return _tsr_model


def remove_background(image_bytes: bytes) -> Image.Image:
    """Remove background from an image, returning an RGBA PIL image."""
    input_image = Image.open(io.BytesIO(image_bytes))
    output_image = remove(input_image)
    # Ensure RGBA
    if output_image.mode != "RGBA":
        output_image = output_image.convert("RGBA")
    return output_image


def generate_mesh(image: Image.Image, output_dir: Path, job_id: str) -> dict:
    """
    Run TripoSR on a clean RGBA image and export mesh files.
    Returns dict with paths to exported files.
    """
    model = get_tsr_model()
    device = next(model.parameters()).device

    # TripoSR expects RGB image on white background
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image_rgb = bg
    else:
        image_rgb = image.convert("RGB")

    # Resize to 256x256 as expected by TripoSR
    image_rgb = image_rgb.resize((256, 256), Image.LANCZOS)

    logger.info(f"[{job_id}] Running TripoSR inference...")
    # Clear CUDA cache before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        scene_codes = model([image_rgb], device=device)

    # Clear VRAM before mesh extraction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"[{job_id}] Extracting mesh...")
    with torch.no_grad():
        meshes = model.extract_mesh(scene_codes, resolution=256, has_vertex_color=True)

    # Clear cache after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_mesh = meshes[0]  # trimesh object

    # Post-process with trimesh
    logger.info(f"[{job_id}] Post-processing mesh...")
    if isinstance(raw_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in raw_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    else:
        mesh = raw_mesh

    # Center and normalize scale
    mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = np.abs(mesh.vertices).max()
    if scale > 0:
        mesh.vertices /= scale

    # Export formats
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # GLB (default)
    glb_path = output_dir / f"{job_id}.glb"
    mesh.export(str(glb_path), file_type="glb")
    results["glb"] = glb_path

    # OBJ
    obj_path = output_dir / f"{job_id}.obj"
    mesh.export(str(obj_path), file_type="obj")
    results["obj"] = obj_path

    # STL
    stl_path = output_dir / f"{job_id}.stl"
    mesh.export(str(stl_path), file_type="stl")
    results["stl"] = stl_path

    logger.info(f"[{job_id}] Mesh exported: {list(results.keys())}")
    return results
