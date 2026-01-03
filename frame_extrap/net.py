import torch
from torch import nn
import torch.nn.functional as F
from utils import forward_warp

def _convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )


def _downsize(x: torch.Tensor):
    return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)


def _mse_loss(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x0, x1, reduction="mean")


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            _convrelu(6, 12, 3, 2, 1), 
            _convrelu(12, 12, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            _convrelu(12, 24, 3, 2, 1), 
            _convrelu(24, 24, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            _convrelu(24, 48, 3, 2, 1), 
            _convrelu(48, 48, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            _convrelu(48, 96, 3, 2, 1), 
            _convrelu(96, 96, 3, 1, 1)
        )
        self.pyramid5 = nn.Sequential(
            _convrelu(96, 192, 3, 2, 1), 
            _convrelu(192, 192, 3, 1, 1)
        )

    def forward(self, img):
        phi1 = self.pyramid1(img)
        phi2 = self.pyramid2(phi1)
        phi3 = self.pyramid3(phi2)
        phi4 = self.pyramid4(phi3)
        phi5 = self.pyramid5(phi4)
        return phi1, phi2, phi3, phi4, phi5


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, side_channels: int):
        super(Decoder, self).__init__()
        self.convblock = nn.Sequential(
            _convrelu(in_channels=in_channels, out_channels=in_channels),
            ResBlock(in_channels=in_channels, side_channels=side_channels),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True)
        )

    def forward(self, phi_0, phi_1, phi_2):
        phi_in = torch.concat([phi_0, phi_1], dim=1)
        if phi_2 is not None:
            phi_in = torch.concat([phi_in, phi_2], dim=1)
        phi_out = self.convblock(phi_in)
        return phi_out[:, :-2, ...], phi_out[:, -2:, ...]


class GFENet(nn.Module):
    """
    Game Flow Extrapolation Net
    """
    def __init__(self) -> None:
        super(GFENet, self).__init__()
        self.encoder = Encoder()
        self.decoder5 = Decoder(192*2, 98, 32)
        self.decoder4 = Decoder(96*3, 50, 32)
        self.decoder3 = Decoder(48*3, 26, 32)
        self.decoder2 = Decoder(24*3, 14, 32)
        self.decoder1 = Decoder(12*3, 8, 32)

    def inference(self, group0: torch.Tensor, group1: torch.Tensor) -> torch.Tensor:
        phi_0_1, phi_0_2, phi_0_3, phi_0_4, phi_0_5 = self.encoder(group0)
        phi_1_1, phi_1_2, phi_1_3, phi_1_4, phi_1_5 = self.encoder(group1)

        phi_2_4, _ = self.decoder5(phi_0_5, phi_1_5, None)
        phi_2_3, _ = self.decoder4(phi_0_4, phi_1_4, phi_2_4)
        phi_2_2, _ = self.decoder3(phi_0_3, phi_1_3, phi_2_3)
        phi_2_1, _ = self.decoder2(phi_0_2, phi_1_2, phi_2_2)
        _,  f_12_0 = self.decoder1(phi_0_1, phi_1_1, phi_2_1)

        return f_12_0

    def forward(self, group0: torch.Tensor, group1: torch.Tensor, group2: torch.Tensor) -> torch.Tensor:
        phi_0_1, phi_0_2, phi_0_3, phi_0_4, phi_0_5 = self.encoder(group0)
        phi_1_1, phi_1_2, phi_1_3, phi_1_4, phi_1_5 = self.encoder(group1)

        GT_1 = _downsize(group2)
        GT_2 = _downsize(GT_1)
        GT_3 = _downsize(GT_2)
        GT_4 = _downsize(GT_3)

        C1_0_gt, C2_0_gt = group2[:, :3, ...], group2[:, 3:6, ...]
        C1_1_gt, C2_1_gt = GT_1[:, :3, ...], GT_1[:, 3:6, ...]
        C1_2_gt, C2_2_gt = GT_2[:, :3, ...], GT_2[:, 3:6, ...]
        C1_3_gt, C2_3_gt = GT_3[:, :3, ...], GT_3[:, 3:6, ...]
        C1_4_gt, C2_4_gt = GT_4[:, :3, ...], GT_4[:, 3:6, ...]

        phi_2_4, f_12_4 = self.decoder5(phi_0_5, phi_1_5, None)
        phi_2_3, f_12_3 = self.decoder4(phi_0_4, phi_1_4, phi_2_4)
        phi_2_2, f_12_2 = self.decoder3(phi_0_3, phi_1_3, phi_2_3)
        phi_2_1, f_12_1 = self.decoder2(phi_0_2, phi_1_2, phi_2_2)
        _, f_12_0 = self.decoder1(phi_0_1, phi_1_1, phi_2_1)

        C2_4_pred = forward_warp(C1_4_gt, f_12_4)
        C2_3_pred = forward_warp(C1_3_gt, f_12_3)
        C2_2_pred = forward_warp(C1_2_gt, f_12_2)
        C2_1_pred = forward_warp(C1_1_gt, f_12_1)
        C2_0_pred = forward_warp(C1_0_gt, f_12_0)

        L_rec_4 = _mse_loss(C2_4_pred, C2_4_gt)
        L_rec_3 = _mse_loss(C2_3_pred, C2_3_gt)
        L_rec_2 = _mse_loss(C2_2_pred, C2_2_gt)
        L_rec_1 = _mse_loss(C2_1_pred, C2_1_gt)
        L_rec_0 = _mse_loss(C2_0_pred, C2_0_gt)

        L_rec = L_rec_0 + L_rec_1 + L_rec_2 + L_rec_3 + L_rec_4

        #TODO: add distillation loss in the future
        loss = L_rec

        return f_12_0, loss

def _self_test_inference():
    group0 = torch.randn(1, 6, 352, 448)
    group1 = torch.randn(1, 6, 352, 448)
    model = GFENet()
    flow_12 = model.inference(group0, group1)
    assert flow_12.shape == (1, 2, 352, 448)

def _self_test_train():
    group0 = torch.randn(1, 6, 352, 448)
    group1 = torch.randn(1, 6, 352, 448)
    group2 = torch.randn(1, 6, 352, 448) # C1,C2 as GT, 3+3=6 channels
    model = GFENet()
    flow_12, loss = model.forward(group0, group1, group2)
    assert flow_12.shape == (1, 2, 352, 448)
    print(f"self_test train loss: {loss.item():.6g}")

if __name__ == "__main__":
    _self_test_inference()
    _self_test_train()
