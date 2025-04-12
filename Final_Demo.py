import torch
import torch.nn as nn
import torch.nn.functional as F

class GRN(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class Conv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.grn = GRN(dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.grn(x)
        return x + residual

def rm(x, mask_ratio=0.5):
    B, C, H, W = x.shape
    num_patches = H * W
    num_keep = int(num_patches * (1 - mask_ratio))

    noise = torch.rand(B, num_patches, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :num_keep]
    x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
    x_masked = torch.gather(x_flat, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C))
    return x_masked, ids_restore

class Convdemo(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.stem = nn.Conv2d(3, dim, kernel_size=4, stride=4)  # Patchify
        self.block = Conv(dim)
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3 * 4 * 4)  # reconstruct 4x4 RGB patch
        )

    def forward(self, x):
        x = self.stem(x)  # [B, dim, H/4, W/4]
        B, C, H, W = x.shape
        x_masked, _ = rm(x, mask_ratio=0.5)  # [B, N_keep, C]
        N = x_masked.shape[1]
        h = int(N**0.5)
        w = N // h
        x_masked = x_masked[:, :h*w, :]  # 자투리 버리기
        x_masked = x_masked.transpose(1, 2).reshape(B, C, h, w)
        x_feat = self.block(x_masked)
        x_feat = x_feat.flatten(2).mean(dim=2)  # [B, dim]
        recon = self.decoder(x_feat)
        return recon.reshape(-1, 3, 4, 4)



if __name__ == "__main__":
    model = Convdemo()
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    print("Reconstructed patch shape:", output.shape)
