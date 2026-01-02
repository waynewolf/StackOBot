import torch


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float()
    y = y.float()
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = (x - mu_x).pow(2).mean()
    sigma_y = (y - mu_y).pow(2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1 = 0.01 * 0.01
    c2 = 0.03 * 0.03
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return numerator / denominator
