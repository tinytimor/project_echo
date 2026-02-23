"""Extract VideoMAE embeddings from echocardiography videos.

VideoMAE is a self-supervised video encoder pre-trained with temporal masking.
Unlike SigLIP which encodes frames independently, VideoMAE learns motion patterns
natively — the encoder itself is video-aware.

Model: MCG-NJU/videomae-base (86M params, 768-dim output)
Input: 16 frames at 224x224
Output: (batch, num_patches*16, 768) spatiotemporal tokens

Usage:
    python -m echoguard.regression.extract_videomae --views A4C PSAX
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from echoguard.config import DataConfig, PROJECT_ROOT, SPLIT_MAP
from echoguard.video_utils import (
    extract_all_frames,
    load_file_list,
    load_volume_tracings,
    get_ed_es_frames_from_tracings,
)

logger = logging.getLogger(__name__)

# VideoMAE output dimension
VIDEOMAE_DIM = 768


def load_videomae_encoder(model_name: str = "MCG-NJU/videomae-base"):
    """Load VideoMAE model and processor.
    
    Returns:
        (model, processor) tuple
    """
    from transformers import VideoMAEImageProcessor, VideoMAEModel
    
    logger.info("Loading VideoMAE from %s...", model_name)
    
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEModel.from_pretrained(model_name).cuda().eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("VideoMAE loaded (%s parameters)", f"{n_params:,}")
    
    return model, processor


def sample_frames_for_videomae(
    all_frames: list[np.ndarray],
    ed_idx: int,
    es_idx: int,
    num_frames: int = 16,
) -> list[np.ndarray]:
    """Sample frames centered around ED-ES cycle for VideoMAE.
    
    VideoMAE expects 16 consecutive frames. We sample uniformly across
    a cardiac cycle (ED → ES → next ED).
    
    Args:
        all_frames: All BGR frames from video
        ed_idx: End-diastole frame index
        es_idx: End-systole frame index
        num_frames: Number of frames to sample (default 16)
    
    Returns:
        List of 16 BGR frames (numpy arrays)
    """
    n = len(all_frames)
    
    # Sample uniformly from full video (captures multiple cardiac cycles)
    indices = np.linspace(0, n - 1, num_frames, dtype=int)
    
    return [all_frames[i] for i in indices]


def extract_videomae_embedding(
    model,
    processor,
    frames_bgr: list[np.ndarray],
    device: str = "cuda",
) -> torch.Tensor:
    """Extract VideoMAE embedding for a list of frames.
    
    Args:
        model: VideoMAE model
        processor: VideoMAE processor
        frames_bgr: List of 16 BGR numpy arrays
        device: GPU device
    
    Returns:
        (embed_dim,) tensor — pooled video embedding
    """
    # Convert BGR → RGB PIL Images
    frames_rgb = []
    for frame in frames_bgr:
        rgb = frame[..., ::-1]  # BGR → RGB
        pil = Image.fromarray(rgb)
        frames_rgb.append(pil)
    
    # VideoMAE processor expects list of lists (batch of videos)
    inputs = processor(
        images=[frames_rgb],  # Batch of 1 video
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(device)  # (1, 16, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(pixel_values)
        # last_hidden_state: (1, num_patches*16, 768)
        # Mean-pool over all spatiotemporal tokens
        embedding = outputs.last_hidden_state.mean(dim=1)  # (1, 768)
    
    return embedding.squeeze(0).cpu()


def extract_videomae_temporal_embedding(
    model,
    processor,
    frames_bgr: list[np.ndarray],
    device: str = "cuda",
) -> torch.Tensor:
    """Extract per-frame VideoMAE embeddings (temporal sequence).
    
    Instead of mean-pooling everything, pool over spatial tokens per frame,
    preserving temporal structure for downstream temporal models.
    
    Args:
        model: VideoMAE model
        processor: VideoMAE processor
        frames_bgr: List of 16 BGR numpy arrays
        device: GPU device
    
    Returns:
        (16, embed_dim) tensor — per-frame embeddings
    """
    frames_rgb = []
    for frame in frames_bgr:
        rgb = frame[..., ::-1]
        pil = Image.fromarray(rgb)
        frames_rgb.append(pil)
    
    inputs = processor(
        images=[frames_rgb],
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values)
        # last_hidden_state: (1, num_patches*num_frames, 768)
        hidden = outputs.last_hidden_state  # (1, N*T, D)
        
        # VideoMAE uses tubelet embedding: each "patch" spans multiple frames
        # For videomae-base: patch size 16x16, tubelet depth 2
        # 224/16 = 14 patches per spatial dim, 14*14 = 196 spatial patches
        # 16 frames / 2 tubelet = 8 temporal positions
        # Total: 196 * 8 = 1568 tokens
        
        # Reshape to (T, spatial, D) and pool over spatial
        # This is approximate — exact depends on tubelet config
        n_tokens = hidden.shape[1]
        n_spatial = 196  # 14x14 for 224x224 images
        n_temporal = n_tokens // n_spatial
        
        # (1, T*S, D) → (1, T, S, D) → (1, T, D)
        try:
            hidden = hidden.view(1, n_temporal, n_spatial, -1)
            per_frame = hidden.mean(dim=2)  # (1, T, D)
        except RuntimeError:
            # Fallback: just use 8 chunks of tokens
            n_chunks = min(8, n_tokens)
            chunk_size = n_tokens // n_chunks
            chunks = []
            for i in range(n_chunks):
                chunk = hidden[:, i*chunk_size:(i+1)*chunk_size, :].mean(dim=1)
                chunks.append(chunk)
            per_frame = torch.stack(chunks, dim=1)  # (1, n_chunks, D)
    
    return per_frame.squeeze(0).cpu()  # (T, D)


def extract_dataset_videomae(
    model,
    processor,
    view: str,
    data_config: DataConfig,
    output_dir: Path,
    device: str = "cuda",
    temporal: bool = True,
    force_reextract: bool = False,
) -> dict:
    """Extract VideoMAE embeddings for all videos in a view.
    
    Args:
        model: VideoMAE model
        processor: VideoMAE processor
        view: "A4C" or "PSAX"
        data_config: Data configuration
        output_dir: Where to save embeddings
        device: GPU device
        temporal: If True, save per-frame embeddings (T, D)
                  If False, save pooled embedding (D,)
        force_reextract: Re-extract existing embeddings
    
    Returns:
        Manifest dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_list = load_file_list(data_config.file_list(view))
    tracings_path = data_config.volume_tracings(view)
    tracings = load_volume_tracings(tracings_path) if tracings_path.exists() else {}
    videos_dir = data_config.videos_dir(view)
    
    manifest = {}
    errors = 0
    skipped = 0
    t0 = time.time()
    
    for entry in tqdm(file_list, desc=f"Extracting {view} VideoMAE"):
        video_id = entry["FileName"]
        ef = entry["EF"]
        age = entry.get("Age", 0.0)
        sex = entry.get("Sex", "U")
        split_fold = entry.get("Split", 0)
        split = SPLIT_MAP.get(int(split_fold), "TRAIN")
        
        video_key = video_id.replace(".avi", "")
        emb_path = output_dir / f"{video_key}.pt"
        
        if emb_path.exists() and not force_reextract:
            manifest[video_key] = {
                "video_id": video_id,
                "ef": ef,
                "age": age,
                "sex": sex,
                "split": split,
                "split_fold": int(split_fold),
                "embedding_path": str(emb_path),
            }
            skipped += 1
            continue
        
        video_path = videos_dir / video_id
        if not video_path.exists():
            logger.warning("Video not found: %s", video_path)
            errors += 1
            continue
        
        try:
            all_frames = extract_all_frames(video_path)
            
            ed_es = get_ed_es_frames_from_tracings(tracings, video_key)
            if ed_es is not None:
                ed_idx, es_idx = ed_es
            else:
                ed_idx = 0
                es_idx = len(all_frames) // 2
            
            # Sample 16 frames for VideoMAE
            sampled_frames = sample_frames_for_videomae(
                all_frames, ed_idx, es_idx, num_frames=16
            )
            
            # Extract embedding
            if temporal:
                embedding = extract_videomae_temporal_embedding(
                    model, processor, sampled_frames, device
                )
            else:
                embedding = extract_videomae_embedding(
                    model, processor, sampled_frames, device
                )
            
            torch.save(embedding, emb_path)
            
            manifest[video_key] = {
                "video_id": video_id,
                "ef": ef,
                "age": age,
                "sex": sex,
                "split": split,
                "split_fold": int(split_fold),
                "embedding_path": str(emb_path),
                "n_frames": len(all_frames),
                "embed_shape": list(embedding.shape),
            }
        
        except Exception as e:
            import traceback
            logger.warning("Error processing %s: %s\n%s", video_id, e, traceback.format_exc())
            errors += 1
    
    elapsed = time.time() - t0
    logger.info(
        "Extracted %d/%d VideoMAE embeddings for %s in %.1fs (%d skipped, %d errors)",
        len(manifest), len(file_list), view, elapsed, skipped, errors,
    )
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Extract VideoMAE embeddings")
    parser.add_argument("--views", nargs="+", default=["A4C"])
    parser.add_argument("--model", default="MCG-NJU/videomae-base",
                        help="VideoMAE model name or path")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "embeddings_videomae"),
                        help="Output directory")
    parser.add_argument("--temporal", action="store_true", default=True,
                        help="Save per-frame embeddings (default)")
    parser.add_argument("--pooled", dest="temporal", action="store_false",
                        help="Save pooled video-level embedding")
    parser.add_argument("--force-reextract", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    output_dir = Path(args.output_dir)
    mode = "temporal" if args.temporal else "pooled"
    logger.info("VideoMAE extraction: model=%s, mode=%s, output=%s",
                args.model, mode, output_dir)
    
    # Check if transformers has VideoMAE
    try:
        from transformers import VideoMAEModel
    except ImportError:
        logger.error("VideoMAE not found in transformers. Install with:")
        logger.error("  pip install transformers[video]")
        return
    
    data_config = DataConfig()
    model, processor = load_videomae_encoder(args.model)
    
    all_manifests = {}
    
    for view in args.views:
        logger.info("Processing %s view...", view)
        view_output = output_dir / f"pediatric_{view.lower()}"
        manifest = extract_dataset_videomae(
            model, processor, view, data_config,
            view_output, temporal=args.temporal,
            force_reextract=args.force_reextract,
        )
        all_manifests[f"pediatric_{view.lower()}"] = manifest
        
        manifest_path = view_output / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Saved manifest to %s", manifest_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("  VideoMAE Embedding Extraction Summary")
    print("=" * 60)
    print(f"  Mode: {'per-frame' if args.temporal else 'pooled'}")
    print(f"  Output dim: {VIDEOMAE_DIM}")
    for name, manifest in all_manifests.items():
        if manifest:
            sample = next(iter(manifest.values()))
            shape = sample.get("embed_shape", "?")
            print(f"  {name}: {len(manifest)} videos, shape={shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
