import argparse
import os
import glob
from typing import Optional, Union, Tuple
import numpy as np
import cv2
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

def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Ensure the input frame is an RGB image.
    If the frame has more than 3 channels, take only the first 3.
    Assumes frame is in shape (H, W, C) and in [0,255].
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected a 3D array for frame, got shape {frame.shape}")
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    return frame

def reduce_channels_to_rgb(pose_image: np.ndarray) -> np.ndarray:
    """
    In case DWpose returns something that is not 3 channels, convert it to an RGB image.
    If the image has a different number of channels (but is 3D), we take the average
    across channels and then stack that into three channels.
    """
    if pose_image.ndim == 3 and pose_image.shape[-1] != 3:
        gray = np.mean(pose_image, axis=-1).clip(0, 255).astype(np.uint8)
        pose_image = np.stack([gray, gray, gray], axis=-1)
    return pose_image

def get_pose_image(frame: np.ndarray) -> np.ndarray:
    """
    Use DWpose directly, returning its neon-on-black skeleton without extra postprocessing.
    Assumes the result is in [0,255] range with a black background and neon lines.
    """
    frame = ensure_rgb(frame)
    raw_pose = pose_detector(frame)      # Pose map from DWpose
    raw_pose = reduce_channels_to_rgb(raw_pose)
    return raw_pose

def encode_pose_sequence(
    vae: AutoencoderKLCausal3D,
    item: ItemInfo,
    bucket_reso: tuple[int, int],
    debug_pose_dir: str = None
) -> torch.Tensor:
    """
    Process the video frames in item.content as follows:
      1) Run get_pose_image() on each frame.
      2) Optionally save each pose image for debugging.
      3) Convert the processed image to a torch tensor (scaled to [-1, 1]).
      4) Stack tensors over the time dimension.
      5) Rearrange dimensions to (B, 3, F, H, W) which is then passed into the VAE.
      6) VAE-encode the pose sequence and return the sampled latent.
    """
    frames = item.content
    if frames.ndim == 3:
        print(f"Single frame detected with shape: {frames.shape}")
        frames = [frames]
    elif frames.ndim == 4:
        print(f"Multiple frames detected with shape: {frames.shape}")
        frames = list(frames)
    else:
        raise ValueError(f"Unsupported shape for item.content: {frames.shape}")

    pose_tensors = []
    for i, frame in enumerate(frames):
        # Produce the skeleton
        pose_img = get_pose_image(frame)
        print(f"Frame {i}: Raw pose image shape: {pose_img.shape}")

        # (A) Optionally save the skeleton image for debugging
        if debug_pose_dir is not None:
            os.makedirs(debug_pose_dir, exist_ok=True)
            # Some item info to ensure unique filenames
            base_stem = os.path.splitext(os.path.basename(item.latent_cache_path))[0]
            debug_fname = f"{base_stem}_pose_{i:03d}.png"
            debug_path = os.path.join(debug_pose_dir, debug_fname)
            Image.fromarray(pose_img.astype(np.uint8)).save(debug_path)
            print(f"Saved debug pose image to {debug_path}")

        # (B) Convert to torch tensor in [-1,1]
        tensor_pose = torch.from_numpy(pose_img).permute(2, 0, 1).contiguous()
        tensor_pose = tensor_pose.to(vae.device, dtype=vae.dtype) / 127.5 - 1.0
        pose_tensors.append(tensor_pose)

    # Stack along new time dimension (F, 3, H, W)
    pose_seq = torch.stack(pose_tensors, dim=0).contiguous()
    print(f"Stacked pose sequence shape (F, 3, H, W): {pose_seq.shape}")

    # Add batch dimension -> (1, F, 3, H, W)
    pose_seq = pose_seq.unsqueeze(0)
    # Rearrange => (1, 3, F, H, W)
    pose_seq = pose_seq.permute(0, 2, 1, 3, 4).contiguous()
    print(f"Pose sequence shape after permuting to (B, 3, F, H, W): {pose_seq.shape}")

    # VAE encode
    with torch.no_grad():
        encoded = vae.encode(pose_seq)
        latent = encoded.latent_dist.sample()
    print(f"Encoded pose latent shape: {latent.shape}")

    return latent

def save_pose_cache(item: ItemInfo, latent: torch.Tensor):
    """
    Save the pose latent to a safetensors file.
    We expect latent to be of shape (1, C, F, H, W) from the VAE,
    then we reformat to (F, C, H, W).
    """
    if latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent.squeeze(0)  # (C, F, H, W)
    # if shape is (C, F, H, W), permute to (F, C, H, W)
    if latent.ndim == 4 and latent.shape[0] == 16 and latent.shape[1] != 16:
        latent = latent.permute(1, 0, 2, 3).contiguous()
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

def encode_and_save_batch(
    vae: AutoencoderKLCausal3D, 
    batch: list[ItemInfo], 
    use_pose: bool = False,
    debug_pose_dir: str = None
):
    # Standard (appearance) latents
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # (B,1,H,W,C)
    contents = contents.permute(0,4,1,2,3).contiguous()  # => (B,C,F,H,W)
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0
    with torch.no_grad():
        latent = vae.encode(contents).latent_dist.sample()
    for item, l in zip(batch, latent):
        save_latent_cache(item, l)

    # Pose latents (if requested)
    if use_pose:
        for item in batch:
            z_pose = encode_pose_sequence(
                vae, 
                item, 
                bucket_reso=item.original_size,
                debug_pose_dir=debug_pose_dir   # Pass the debug folder path
            )
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

    # Load VAE
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

    # Encode
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
                encode_and_save_batch(
                    vae,
                    batch[j : j + bs],
                    use_pose=args.use_pose,
                    debug_pose_dir=args.debug_pose_dir  # pass debug folder
                )

        # Remove cache files no longer needed
        all_latent_cache_paths = [os.path.normpath(p) for p in all_latent_cache_paths]
        all_latent_cache_paths = set(all_latent_cache_paths)
        all_cache_files = dataset.get_all_latent_cache_files()
        for cache_file in all_cache_files:
            if os.path.normpath(cache_file) not in all_latent_cache_paths:
                if args.keep_cache:
                    logger.info(f"Keeping old cache file: {cache_file}")
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
    parser.add_argument("--use_pose", action="store_true", help="also cache pose latent from video using DWpose")

    # (A) Add an argument for the debug pose folder
    parser.add_argument(
        "--debug_pose_dir",
        type=str,
        default=None,
        help="Folder path where to save skeleton pose images for debugging."
    )

    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)