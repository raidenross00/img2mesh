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
from tsr.bake_texture import make_atlas, rasterize_position_atlas
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

    # Bake texture: UV unwrap with xatlas, rasterize position atlas with moderngl,
    # then query triplane on GPU for colors (keeps model on GPU for correct results)
    logger.info(f"[{job_id}] Baking texture...")
    texture_resolution = 2048
    texture_padding = round(max(2, texture_resolution / 256))

    atlas = make_atlas(mesh, texture_resolution, texture_padding)
    positions_texture = rasterize_position_atlas(
        mesh, atlas["vmapping"], atlas["indices"], atlas["uvs"],
        texture_resolution, texture_padding,
    )

    # Query colors on GPU (keeps model on correct device)
    scene_code = scene_codes[0]
    positions = torch.tensor(
        positions_texture.reshape(-1, 4)[:, :3],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        queried = model.renderer.query_triplane(model.decoder, positions, scene_code)
    rgb_f = queried["color"].cpu().numpy().reshape(-1, 3)
    alpha_mask = positions_texture.reshape(-1, 4)[:, 3]
    rgba_f = np.concatenate([rgb_f, alpha_mask[:, None]], axis=1)
    rgba_f[alpha_mask == 0.0] = [0, 0, 0, 0]
    colors = rgba_f.reshape(texture_resolution, texture_resolution, 4)

    bake_output = {
        "vmapping": atlas["vmapping"],
        "indices": atlas["indices"],
        "uvs": atlas["uvs"],
        "colors": colors,
    }
    logger.info(f"[{job_id}] Texture baking complete.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Export formats
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Build the texture image — convert RGBA to RGB (fill empty areas with black)
    colors = bake_output["colors"]
    logger.info(f"[{job_id}] Texture colors shape={colors.shape}, min={colors.min():.3f}, max={colors.max():.3f}")
    # Log color stats for non-transparent pixels only
    alpha = colors[:, :, 3]
    visible = colors[alpha > 0]
    if visible.size > 0:
        logger.info(f"[{job_id}] Visible texel RGB min={visible[:, :3].min():.3f}, max={visible[:, :3].max():.3f}, mean={visible[:, :3].mean():.3f}")
    # Convert to RGB — use black for empty texels
    rgb = (colors[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
    texture_img = Image.fromarray(rgb).transpose(Image.FLIP_TOP_BOTTOM)
    texture_path = output_dir / f"{job_id}_texture.png"
    texture_img.save(str(texture_path))
    logger.info(f"[{job_id}] Saved texture to {texture_path} ({texture_img.size})")

    # Get remapped mesh data
    vmapping = bake_output["vmapping"]
    indices = bake_output["indices"]
    uvs = bake_output["uvs"]

    # OBJ with texture
    logger.info(f"[{job_id}] Exporting OBJ with texture...")
    obj_path = output_dir / f"{job_id}.obj"
    xatlas.export(
        str(obj_path),
        mesh.vertices[vmapping],
        indices,
        uvs,
        mesh.vertex_normals[vmapping],
    )
    mtl_path = output_dir / f"{job_id}.mtl"
    mtl_path.write_text(
        f"newmtl material0\n"
        f"map_Kd {job_id}_texture.png\n"
    )
    obj_content = obj_path.read_text()
    obj_path.write_text(f"mtllib {job_id}.mtl\n{obj_content}")
    results["obj"] = obj_path

    # GLB with texture
    logger.info(f"[{job_id}] Exporting GLB with baked texture...")
    glb_path = output_dir / f"{job_id}.glb"
    textured_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[vmapping],
        faces=indices,
        process=False,
    )
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_img,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    textured_mesh.visual = trimesh.visual.TextureVisuals(
        uv=uvs,
        material=material,
    )
    textured_mesh.export(str(glb_path), file_type="glb")
    logger.info(f"[{job_id}] GLB exported ({glb_path.stat().st_size} bytes)")
    results["glb"] = glb_path

    # STL (no color support)
    stl_path = output_dir / f"{job_id}.stl"
    textured_mesh.export(str(stl_path), file_type="stl")
    results["stl"] = stl_path

    logger.info(f"[{job_id}] Mesh exported: {list(results.keys())}")
    return results
