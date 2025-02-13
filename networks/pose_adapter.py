import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseAdapter(nn.Module):
    """
    Example pose adapter that takes (B, inC, T, H, W) 
    and outputs (B, outC, T, H, W) for injection.
    """
    def __init__(self, in_channels=16, out_channels=16, mid_channels=32, num_layers=3):
        super().__init__()
        self.out_channels = out_channels
        layers = []
        # Example pipeline: strided conv3d or repeated conv to minimize dimension
        layers.append(nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, inC, T, H, W)
        return self.net(x)


class DiffusionTransformerWithPose(nn.Module):
    """
    Wraps an existing "base_transformer" (which does the patch embedding, block attention,
    etc.), plus a PoseAdapter. In the forward pass, we:
      1) Patch-embed x.
      2) Interpolate & project pose_input to match patch embedding’s T,H,W.
      3) Insert pose features at selected injection_layers.
    """

    def __init__(self, base_transformer, pose_adapter: PoseAdapter, injection_layers=(2, 5, 8)):
        super().__init__()
        # The base model is your pre-existing HYVideoDiffusionTransformer,
        # which has .img_in, .double_blocks, .single_blocks, .final_layer, etc.
        self.transformer = base_transformer
        self.pose_adapter = pose_adapter
        self.injection_layers = set(injection_layers)  # layers to apply pose injection

        # A simple 1×1×1 conv (or linear) to match your hidden dimension after pose_adapter
        # so that pose_features match self.transformer.hidden_size.
        # If hidden_size=3072, you typically flatten in HxW, so we do a 3D→2D projection eventually.
        self.pose_proj = nn.Conv3d(pose_adapter.out_channels, self.transformer.hidden_size, kernel_size=1)
        # If the base model sometimes has 4096 as the final hidden, you may want a reduce_proj, etc.
        # Shown here only if needed:
        # self.reduce_proj = nn.Linear(4096, 3072)

    def forward(
        self,
        x: torch.Tensor,               # (B, inC, T_orig, H_orig, W_orig)
        t: torch.Tensor,               # (B, ) timesteps
        text_states: torch.Tensor,     # (B, textLen, hiddenDim1)
        pose_input: torch.Tensor=None, # (B, 16, T_orig, H_orig, W_orig) or something
        pose_alpha: float=1.0,
        **kwargs
    ):
        """
        1) run patch embed:  x -> (B, hidden_size, T, H, W)
        2) flatten -> (B, T×H×W, hidden_size)
        3) run double_blocks in separate image & text streams
        4) optionally add pose block outputs after certain double_blocks
        5) merge & run single_blocks
        6) final layer -> up or unpatchify
        """

        # 1) Image patch embedding (the base model's standard approach).
        #    Suppose base_transformer.img_in(...) => shape (B, hidden_size, T, H, W).
        img = self.transformer.img_in(x)  
        if img.dim() != 5:
            raise ValueError(f"Expected (B, hidden_size, T, H, W), got shape {img.shape}")

        B, hidden_size, T, H, W = img.shape

        # Flatten from (B, hidden_size, T, H, W) to (B, T×H×W, hidden_size).
        # Our base HYVideoDiffusionTransformer does something like that internally,
        # but let's do it explicitly so we can see the shape:
        img = img.permute(0, 2, 3, 4, 1).contiguous()  # => (B, T, H, W, hidden_size)
        img = img.view(B, T*H*W, hidden_size)          # => (B, T×H×W, hidden_size)

        # 2) If we have a pose_input, pass it to the pose_adapter, then
        #    interpolate to the exact (T, H, W).
        if pose_input is not None:
            # Pose adapter: shape => (B, outC, T_orig, H_orig, W_orig) => (B, outC, T~, H~, W~)
            pose_features = self.pose_adapter(pose_input)  # still (B, outC, T_orig, H_orig, W_orig)

            # Interpolate to the same T, H, W used by the patch embed:
            pose_features_ds = F.interpolate(
                pose_features,
                size=(T, H, W),       # must match the shape from patch embed
                mode="trilinear",
                align_corners=False,
            )
            # Now shape = (B, outC, T, H, W).

            # Project to match the transformer's hidden_size:
            pose_features_proj = self.pose_proj(pose_features_ds)
            # => (B, hidden_size, T, H, W)

            # Flatten to the same shape as “img”: (B, T×H×W, hidden_size)
            pose_features_proj = pose_features_proj.permute(0, 2, 3, 4, 1).contiguous()
            pose_features_proj_flat = pose_features_proj.view(B, T*H*W, hidden_size)
        else:
            pose_features_proj_flat = None

        # 3) Now we run the double_blocks on separate image & text streams:
        #    our base transformer typically unflattens them into (B, #tokens, hidden_size).
        #    so let's do that next:
        # T×H×W might be "img_seq_len", for example.
        img_seq_len = img.shape[1]

        # For text, e.g. shape (B, textLen, hiddenDim1).
        txt = self.transformer.txt_in(text_states)  # or single_refiner, etc.
        txt_seq_len = txt.shape[1]

        # The base model’s code typically merges the vector t or text_states_2 into a “conditioning vector” 
        # so we do it similarly here. For brevity, we’ll just define:
        vec = self.transformer.time_in(t)

        # 4) For each double_blocks i, run block; optionally inject pose at i.
        #    The base_transformer double blocks take (img, txt, vec...) as well.
        for i, block in enumerate(self.transformer.double_blocks):
            # unflatten "img" if necessary
            # but typically the base code wants (B, L, hidden_size), so we already have that shape: (B, T×H×W, hidden_size)
            img_out, txt_out = block(img, txt, vec)

            img, txt = img_out, txt_out

            # Pose injection:
            if (pose_features_proj_flat is not None) and (i in self.injection_layers):
                # “Add” pose:
                # shape is (B, T×H×W, hidden_size)
                # Make sure it’s the same length in dimension 1:
                if pose_features_proj_flat.shape[1] != img.shape[1]:
                    raise ValueError(f"Mismatch: pose tokens={pose_features_proj_flat.shape[1]}, "
                                     f"img tokens={img.shape[1]}. Need same shape.")
                img = img + pose_alpha * pose_features_proj_flat

        # 5) Merge “img” + “txt” tokens => single stream
        x_merged = torch.cat([img, txt], dim=1)  # shape (B, (img_seq_len+txt_seq_len), hidden_size)

        # pass through single blocks:
        for block in self.transformer.single_blocks:
            x_merged = block(x_merged, vec, txt_seq_len)

        # Now the image tokens are x_merged[:, :img_seq_len, :]
        output_img = x_merged[:, :img_seq_len, :]

        # 6) If the base model does a final linear or final up-projection:
        out = self.transformer.final_layer(output_img, vec)
        # Possibly unpatchify if your pipeline expects (B, outC, T_orig, H_orig, W_orig) at the end:
        out = self.transformer.unpatchify(out, T, H, W)
        return out