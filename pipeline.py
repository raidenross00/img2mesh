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


def _prepare_image(image: Image.Image, foreground_ratio: float = 0.85) -> Image.Image:
    """
    Preprocess image for TripoSR: crop foreground, pad to square,
    add margin, composite onto gray background. Matches the official pipeline.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    arr = np.array(image)
    alpha = np.where(arr[..., 3] > 0)
    if alpha[0].size == 0:
        # No foreground found, return gray square
        return Image.new("RGB", (256, 256), (127, 127, 127))

    # Crop to foreground bounding box
    y1, y2 = alpha[0].min(), alpha[0].max()
    x1, x2 = alpha[1].min(), alpha[1].max()
    fg = arr[y1:y2, x1:x2]

    # Pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    padded = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=0)

    # Add margin so foreground occupies foreground_ratio of the frame
    new_size = int(padded.shape[0] / foreground_ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    padded = np.pad(padded, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=0)

    # Alpha-composite onto gray (0.5) background
    img_f = padded.astype(np.float32) / 255.0
    alpha_mask = img_f[..., 3:4]
    rgb = img_f[..., :3] * alpha_mask + (1 - alpha_mask) * 0.5
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    result = Image.fromarray(rgb).resize((256, 256), Image.LANCZOS)
    return result


def generate_mesh(image: Image.Image, output_dir: Path, job_id: str) -> dict:
    """
    Run TripoSR on a clean RGBA image and export mesh files.
    Returns dict with paths to exported files.
    """
    model = get_tsr_model()
    device = next(model.parameters()).device

    image_rgb = _prepare_image(image)

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

    # Ensure vertex colors are uint8 [0,255] — TripoSR returns float [0,1]
    if mesh.visual.vertex_colors is not None:
        vc = np.array(mesh.visual.vertex_colors, dtype=np.float64)
        if vc.max() <= 1.0:
            vc = (vc * 255).clip(0, 255).astype(np.uint8)
        else:
            vc = vc.clip(0, 255).astype(np.uint8)
        # Ensure RGBA
        if vc.shape[-1] == 3:
            alpha = np.full((vc.shape[0], 1), 255, dtype=np.uint8)
            vc = np.hstack([vc, alpha])
        mesh.visual.vertex_colors = vc

    # Center and normalize scale
    mesh.vertices -= mesh.vertices.mean(axis=0)
    scale = np.abs(mesh.vertices).max()
    if scale > 0:
        mesh.vertices /= scale

    # Export formats
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # GLB (default — supports vertex colors natively)
    glb_path = output_dir / f"{job_id}.glb"
    mesh.export(str(glb_path), file_type="glb")
    results["glb"] = glb_path

    # OBJ with material — bake vertex colors into a texture for OBJ support
    obj_path = output_dir / f"{job_id}.obj"
    try:
        # Convert vertex colors to a face-based texture via trimesh
        textured = mesh.copy()
        textured.visual = textured.visual.to_texture()
        textured.export(str(obj_path), file_type="obj")
    except Exception:
        # Fallback: export without texture
        mesh.export(str(obj_path), file_type="obj")
    results["obj"] = obj_path

    # STL (no color support)
    stl_path = output_dir / f"{job_id}.stl"
    mesh.export(str(stl_path), file_type="stl")
    results["stl"] = stl_path

    logger.info(f"[{job_id}] Mesh exported: {list(results.keys())}")
    return results
