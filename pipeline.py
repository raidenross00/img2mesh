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
import xatlas
from PIL import Image
from tsr.bake_texture import bake_texture
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
    Uses texture baking for proper colors, matching official run.py --bake-texture.
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

    # Extract mesh WITHOUT vertex colors (bake_texture handles colors separately)
    logger.info(f"[{job_id}] Extracting mesh...")
    with torch.no_grad():
        meshes = model.extract_mesh(scene_codes, has_vertex_color=False, resolution=256)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mesh = meshes[0]

    # Bake texture atlas — queries the triplane directly for high-quality colors
    logger.info(f"[{job_id}] Baking texture...")
    bake_output = bake_texture(mesh, model, scene_codes[0], texture_resolution=2048)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Export formats
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Build the texture image
    texture_img = Image.fromarray(
        (bake_output["colors"] * 255.0).astype(np.uint8)
    ).transpose(Image.FLIP_TOP_BOTTOM)
    texture_path = output_dir / f"{job_id}_texture.png"
    texture_img.save(str(texture_path))

    # Get remapped mesh data
    vmapping = bake_output["vmapping"]
    indices = bake_output["indices"]
    uvs = bake_output["uvs"]

    # OBJ with texture (primary format — best color support)
    logger.info(f"[{job_id}] Exporting OBJ with texture...")
    obj_path = output_dir / f"{job_id}.obj"
    xatlas.export(
        str(obj_path),
        mesh.vertices[vmapping],
        indices,
        uvs,
        mesh.vertex_normals[vmapping],
    )
    # Write MTL file referencing the texture
    mtl_path = output_dir / f"{job_id}.mtl"
    mtl_path.write_text(
        f"newmtl material0\n"
        f"map_Kd {job_id}_texture.png\n"
    )
    # Prepend mtllib to OBJ
    obj_content = obj_path.read_text()
    obj_path.write_text(f"mtllib {job_id}.mtl\n{obj_content}")
    results["obj"] = obj_path

    # GLB with texture
    logger.info(f"[{job_id}] Exporting GLB...")
    glb_path = output_dir / f"{job_id}.glb"
    textured_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[vmapping],
        faces=indices,
        process=False,
    )
    textured_mesh.visual = trimesh.visual.TextureVisuals(
        uv=uvs,
        image=texture_img,
    )
    textured_mesh.export(str(glb_path), file_type="glb")
    results["glb"] = glb_path

    # STL (no color support)
    stl_path = output_dir / f"{job_id}.stl"
    textured_mesh.export(str(stl_path), file_type="stl")
    results["stl"] = stl_path

    logger.info(f"[{job_id}] Mesh exported: {list(results.keys())}")
    return results
