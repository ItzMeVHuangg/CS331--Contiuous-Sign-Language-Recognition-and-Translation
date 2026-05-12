import torch.nn.functional as F

def enforce_target_len(x, target_len):
    """
    x: [B, T, C]
    """
    B, T, C = x.shape
    if T == target_len:
        return x

    x = x.permute(0, 2, 1)  # [B, C, T]
    x = F.interpolate(
        x,
        size=target_len,
        mode="linear",
        align_corners=False
    )
    x = x.permute(0, 2, 1)  # [B, T, C]
    return x