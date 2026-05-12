
import torch
import torch.nn as nn


class CSLRModel(nn.Module):
    """
    Full CSLR encoder: CNN (2-D ResNet-18 or 3-D R3D-18) → BiLSTM-CTC.

    forward() returns:
        log_probs : (T, B, num_classes)   — CTC log-probabilities
        hidden    : (B, T, proj_dim)      — projected features for SLT fusion
        adj_lens  : (B,)                  — frame lengths adjusted for any
                                            temporal downsampling by the CNN
    """

    def __init__(self, cfg: dict, num_classes: int, use_3d_cnn: bool = False, use_mediapipe: bool = False):
        super().__init__()

        if use_mediapipe:
            from models.mediapipe_encoder import MediaPipeEncoder
            self.cnn = MediaPipeEncoder(out_features=cfg["mediapipe"]["out_features"])
            input_size = cfg["mediapipe"]["out_features"]
        elif use_3d_cnn:
            from models.cnn_encoder_3d import CNNEncoder3D
            self.cnn = CNNEncoder3D(
                pretrained   = cfg["cnn_3d"]["pretrained"],
                out_features = cfg["cnn_3d"]["out_features"],
                clip_len     = cfg.get("cnn_3d", {}).get("clip_len", 8),
                clip_stride  = cfg.get("cnn_3d", {}).get("clip_stride", None),
            )
            input_size = cfg["cnn_3d"]["out_features"]
        else:
            from models.cnn_encoder import CNNEncoder
            self.cnn = CNNEncoder(
                pretrained   = cfg["cnn"]["pretrained"],
                out_features = cfg["cnn"]["out_features"],
                freeze_bn    = cfg["cnn"]["freeze_bn"],
            )
            input_size = cfg["cnn"]["out_features"]

        from models.bilstm_ctc import BiLSTM_CTC
        self.bilstm_ctc = BiLSTM_CTC(
            input_size      = input_size,
            hidden_size     = cfg["bilstm"]["hidden_size"],
            num_layers      = cfg["bilstm"]["num_layers"],
            num_classes     = num_classes,
            dropout         = cfg["bilstm"]["dropout"],
            projection_size = cfg["bilstm"]["projection_size"],
            blank_idx       = cfg["cslr"]["ctc_blank_idx"],
        )

    def forward(
        self,
        frames: torch.Tensor,      # (B, T, C, H, W)
        frame_lens: torch.Tensor,  # (B,)
        keypoints: torch.Tensor = None, # (B, T, 225)
    ):
        from models.mediapipe_encoder import MediaPipeEncoder
        if isinstance(self.cnn, MediaPipeEncoder) and keypoints is not None:
            feats = self.cnn(keypoints)
        else:
            feats = self.cnn(frames)          # (B, T', feat_dim)
        T_out = feats.shape[1]

        # Proportionally scale lengths if CNN reduced temporal resolution
        if T_out < frame_lens.max().item():
            scale    = T_out / frame_lens.float().max().item()
            adj_lens = (frame_lens.float() * scale).long().clamp(min=1, max=T_out)
        else:
            adj_lens = frame_lens.clamp(max=T_out)

        log_probs, hidden = self.bilstm_ctc(feats, adj_lens)
        return log_probs, hidden, adj_lens
