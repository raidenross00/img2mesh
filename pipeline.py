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
import rembg
import torch
import trimesh
from PIL import Image
from tsr.utils import remove_background as tsr_remove_background, resize_foreground

logger = logging.getLogger(__name__)

# Global model reference — loaded once on first use
_tsr_model = None
_rembg_session = None


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
        _tsr_model.renderer.set_chunk_size(8192)
        _tsr_model.to(device)
        logger.info(f"TripoSR model loaded on {device}")
    return _tsr_model


def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = rembg.new_session()
    return _rembg_session


def remove_background(image_bytes: bytes) -> Image.Image:
    """Remove background using the official TripoSR approach."""
    input_image = Image.open(io.BytesIO(image_bytes))
    return tsr_remove_background(input_image, get_rembg_session())


def prepare_image(image: Image.Image, foreground_ratio: float = 0.85) -> Image.Image:
    """
    Preprocess RGBA image for TripoSR, matching the official run.py pipeline exactly:
    resize_foreground -> float normalize -> alpha composite on gray -> uint8 PIL Image.
    """
    image = resize_foreground(image, foreground_ratio)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def generate_mesh(image: Image.Image, output_dir: Path, job_id: str) -> dict:
    """
    Run TripoSR on a clean RGBA image and export mesh files.
    Returns dict with paths to exported files.
    """
    model = get_tsr_model()
    device = next(model.parameters()).device

    image_rgb = prepare_image(image)

    logger.info(f"[{job_id}] Running TripoSR inference...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        scene_codes = model([image_rgb], device=device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"[{job_id}] Extracting mesh...")
    with torch.no_grad():
        meshes = model.extract_mesh(scene_codes, resolution=256, has_vertex_color=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mesh = meshes[0]

    # Post-process with trimesh
    logger.info(f"[{job_id}] Post-processing mesh...")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    # Center and normalize scale
    mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = np.abs(mesh.vertices).max()
    if scale > 0:
        mesh.vertices /= scale

    # Export formats
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # GLB (supports vertex colors natively)
    glb_path = output_dir / f"{job_id}.glb"
    mesh.export(str(glb_path), file_type="glb")
    results["glb"] = glb_path

    # OBJ (vertex colors baked to texture)
    obj_path = output_dir / f"{job_id}.obj"
    try:
        textured = mesh.copy()
        textured.visual = textured.visual.to_texture()
        textured.export(str(obj_path), file_type="obj")
    except Exception:
        mesh.export(str(obj_path), file_type="obj")
    results["obj"] = obj_path

    # STL (no color support)
    stl_path = output_dir / f"{job_id}.stl"
    mesh.export(str(stl_path), file_type="stl")
    results["stl"] = stl_path

    logger.info(f"[{job_id}] Mesh exported: {list(results.keys())}")
    return results
