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
from tsr.utils import scale_tensor

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


def _extract_mesh_low_vram(
    model, scene_code, resolution: int = 256, threshold: float = 25.0, chunk_size: int = 50000
) -> trimesh.Trimesh:
    """
    Extract mesh from scene code using chunked GPU queries to stay within VRAM.
    Density and color are queried in small batches, marching cubes runs on CPU.
    """
    model.set_marching_cubes_resolution(resolution)
    helper = model.isosurface_helper
    grid_verts = helper.grid_vertices  # (resolution^3, 3) on CPU or GPU
    radius = model.renderer.cfg.radius

    # Scale grid vertices to model space
    scaled_verts = scale_tensor(
        grid_verts, helper.points_range, (-radius, radius)
    )

    # Query density in chunks on GPU
    density_chunks = []
    for i in range(0, scaled_verts.shape[0], chunk_size):
        chunk = scaled_verts[i : i + chunk_size].to(scene_code.device)
        with torch.no_grad():
            d = model.renderer.query_triplane(model.decoder, chunk, scene_code)["density_act"]
        density_chunks.append(d.cpu())
        del chunk, d
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    density = torch.cat(density_chunks, dim=0)

    # Run marching cubes on CPU
    v_pos, t_pos_idx = helper(-(density.to(helper.grid_vertices.device) - threshold))
    v_pos = scale_tensor(v_pos, helper.points_range, (-radius, radius))

    # Query vertex colors in chunks on GPU
    color_chunks = []
    v_pos_gpu = v_pos.to(scene_code.device)
    for i in range(0, v_pos_gpu.shape[0], chunk_size):
        chunk = v_pos_gpu[i : i + chunk_size]
        with torch.no_grad():
            c = model.renderer.query_triplane(model.decoder, chunk, scene_code)["color"]
        color_chunks.append(c.cpu())
        del chunk, c
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    color = torch.cat(color_chunks, dim=0)

    return trimesh.Trimesh(
        vertices=v_pos.cpu().numpy(),
        faces=t_pos_idx.cpu().numpy(),
        vertex_colors=color.numpy(),
    )


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

    # Extract mesh with chunked GPU queries to avoid OOM on 8GB cards
    logger.info(f"[{job_id}] Extracting mesh (low-VRAM mode)...")
    raw_mesh = _extract_mesh_low_vram(model, scene_codes[0], resolution=256, chunk_size=50000)

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
