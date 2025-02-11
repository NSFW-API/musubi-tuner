import argparse
import os
import glob
from typing import Optional, Union, Tuple
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import logging
from safetensors.torch import save_file

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from dataset.image_video_dataset import BaseDataset, ItemInfo, save_latent_cache
from hunyuan_model.vae import load_vae
from hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from utils.model_utils import str_to_dtype

# Import your pose detector (DWpose) from the annotator folder.
from annotator.dwpose import DWposeDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the DWpose detector once.
pose_detector = DWposeDetector()

def get_pose_image(frame: np.ndarray) -> np.ndarray:
    """
    Given a single frame, return the pose-processed image.
    If the input frame has an extra batch dimension (shape (1, H, W, C)), squeeze it.
    """
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = np.squeeze(frame, axis=0)
    return pose_detector(frame)

def encode_pose_sequence(vae: AutoencoderKLCausal3D, item: ItemInfo, bucket_reso: Tuple[int, int]) -> torch.Tensor:
    """
    Given item.content (a numpy array with shape (F, H, W, C) for a multi‑frame video or (H, W, C) for a single image),
    apply DWposeDetector on each frame, stack the results, and reformat the tensor for the VAE.
    Returns a tensor of pose images with final shape (1, C, F, H, W).
    """
    frames = item.content
    # If item.content is a single image, wrap it in a list
    if frames.ndim == 3:
        frames = [frames]
    elif frames.ndim == 4:
        frames = list(frames)
    else:
        raise ValueError(f"Unsupported shape for item.content: {frames.shape}")
    
    pose_frames = []
    for frame in frames:
        # Make sure each frame is of shape (H, W, C)
        processed = get_pose_image(frame)  # returns a numpy array (H, W, C)
        pose_frames.append(torch.from_numpy(processed))
    
    # Stack along time: shape (F, H, W, C)
    pose_seq = torch.stack(pose_frames, dim=0)
    # Add batch dimension: becomes (1, F, H, W, C)
    pose_seq = pose_seq.unsqueeze(0)
    # Permute to get shape (1, C, F, H, W)
    pose_seq = pose_seq.permute(0, 4, 1, 2, 3).contiguous()
    # Move to same device and dtype as the VAE; normalize [-1,1]
    pose_seq = pose_seq.to(vae.device, dtype=vae.dtype)
    pose_seq = pose_seq / 127.5 - 1.0
    return pose_seq

def save_pose_cache(item: ItemInfo, latent: torch.Tensor):
    """
    Save the pose latent to a safetensors file.
    We expect latent to be of shape (1, C, F, H, W) from the VAE.
    We then squeeze the batch dimension and, if necessary, permute to get (F, C, H, W)
    """
    if latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent.squeeze(0)  # now (C, F, H, W)
    # We expect the final saved tensor to have frame as first dimension, so if shape is (C, F, H, W), permute:
    expected_channels = 16  # update as needed (from VAE latent channels)
    if latent.ndim == 4 and latent.shape[0] != expected_channels and latent.shape[1] == expected_channels:
        latent = latent.permute(1, 0, 2, 3).contiguous()  # now (F, C, H, W)
    assert latent.ndim == 4, "latent should be 4D tensor (frame, channel, height, width)"
    folder = os.path.dirname(item.latent_cache_path)
    base = os.path.splitext(os.path.basename(item.latent_cache_path))[0]
    pose_cache_path = os.path.join(folder, base + "_pose.safetensors")
    metadata = {
        "architecture": "hunyuan_video_pose",
        "width": f"{item.original_size[0]}",
        "height": f"{item.original_size[1]}",
        "format_version": "1.0.0",
    }
    save_file({"latent": latent.detach().cpu()}, pose_cache_path, metadata=metadata)
    logger.info(f"Saved pose latent cache to: {pose_cache_path}")

def encode_and_save_batch(vae: AutoencoderKLCausal3D, batch: list[ItemInfo], use_pose: bool = False):
    # Process the existing video latent as in your original code:
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        # if content shape is (B, H, W, C), add a frame dimension:
        contents = contents.unsqueeze(1)
    # Rearrange to (B, C, F, H, W)
    contents = contents.permute(0, 4, 1, 2, 3).contiguous()
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0
    with torch.no_grad():
        latent = vae.encode(contents).latent_dist.sample()
    for item, l in zip(batch, latent):
        save_latent_cache(item, l)
    
    # NEW: if use_pose flag is set, process the pose branch:
    if use_pose:
        for item in batch:
            pose_seq = encode_pose_sequence(vae, item, bucket_reso=item.original_size)
            with torch.no_grad():
                z_pose = vae.encode(pose_seq).latent_dist.sample()
            save_pose_cache(item, z_pose)

def main(args):
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    # Debug mode
    if args.debug_mode is not None:
        from dataset.image_video_dataset import show_datasets
        show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    assert args.vae is not None, "vae checkpoint is required"

    # Load VAE model (HunyuanVideo VAE – which is float16)
    vae_dtype = torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device=device, vae_path=args.vae)
    vae.eval()
    logger.info(f"Loaded VAE: {vae.config}, dtype: {vae.dtype}")

    if args.vae_chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
        logger.info(f"Set chunk_size to {args.vae_chunk_size} for CausalConv3d in VAE")
    if args.vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)
        vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
        vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
    elif args.vae_tiling:
        vae.enable_spatial_tiling(True)

    # Encode each dataset sample
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_latent_cache_paths = []
        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            all_latent_cache_paths.extend([item.latent_cache_path for item in batch])
            if args.skip_existing:
                filtered_batch = [item for item in batch if not os.path.exists(item.latent_cache_path)]
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch
            bs = args.batch_size if args.batch_size is not None else len(batch)
            for j in range(0, len(batch), bs):
                encode_and_save_batch(vae, batch[j : j + bs], use_pose=args.use_pose)
        # Remove cache files no longer needed (existing code as is)...
        all_latent_cache_paths = [os.path.normpath(p) for p in all_latent_cache_paths]
        all_latent_cache_paths = set(all_latent_cache_paths)
        all_cache_files = dataset.get_all_latent_cache_files()
        for cache_file in all_cache_files:
            if os.path.normpath(cache_file) not in all_latent_cache_paths:
                if args.keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--vae", type=str, required=False, default=None, help="path to vae checkpoint")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is float16")
    parser.add_argument("--vae_tiling", action="store_true", help="enable spatial tiling for VAE")
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument("--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256")
    parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size; overrides dataset config if dataset batch size > this")
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset; default is cpu count-1")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
    parser.add_argument("--debug_mode", type=str, default=None, choices=["image", "console"], help="debug mode")
    parser.add_argument("--console_width", type=int, default=80, help="debug mode: console width")
    parser.add_argument("--console_back", type=str, default=None, help="debug mode: console background; choice from ascii_magic.Back")
    parser.add_argument("--console_num_images", type=int, default=None, help="debug mode: number of images to show for each dataset")
    # Add new flag for pose caching:
    parser.add_argument("--use_pose", action="store_true", help="also cache pose latent from video using DWpose")
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)