
# ConvNeXt V2 based Fully Convolutional Masked Autoencoder (FC-MAE) with GRN on CIFAR-10

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Global Response Normalization module from ConvNeXt V2
class GRN(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        # Learnable scale and bias, initialized to zero as in ConvNeXt V2
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: Tensor of shape (B, C, H, W)
        # 1. Compute L2 norm for each channel (across spatial H, W)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # shape (B, C, 1, 1)
        # 2. Normalize the norms by their mean across channels (feature competition)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)  # shape (B, C, 1, 1)
        # 3. Calibrate input using normalized values, and apply affine transform (gamma, beta)
        return self.gamma * (x * nx) + self.beta + x

# ConvNeXt block with depthwise conv, pointwise convs, and GRN (ConvNeXt V2 style)
class CNB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Depthwise convolution (spatial convolution for each channel separately)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # Pointwise layers and activation
        self.norm = nn.BatchNorm2d(dim)           # Using BatchNorm for simplicity (original ConvNeXt uses LayerNorm)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)  # expansion
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)  # projection back to dim
        # Global Response Normalization after the MLP conv layers
        self.grn = GRN(dim)

    def forward(self, x):
        # ConvNeXt V2 block: depthwise conv -> BN -> 1x1 conv -> GELU -> 1x1 conv -> GRN -> residual add
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.grn(x)
        return x + residual  # residual connection

# ConvNeXt V2 Autoencoder model (FC-MAE: Fully Convolutional Masked Autoencoder)
class CNA(nn.Module):
    def __init__(self, dim=32, enc_blocks=2):
        """
        dim: base channel dimension for patch embeddings and conv blocks.
        enc_blocks: number of ConvNeXt blocks in the encoder.
        The decoder uses one ConvNeXt block and a final conv to reconstruct the image.
        """
        super().__init__()
        # Patchify layer: split image into non-overlapping patches (here 4x4 patches via a stride-4 conv)
        # This reduces 32x32 image with 3 channels to (dim) feature maps of size 8x8 (32/4 = 8 patches per side).
        self.stem = nn.Conv2d(3, dim, kernel_size=4, stride=4)

        # Encoder: stack of ConvNeXt blocks with GRN
        self.enc_blocks = nn.ModuleList([CNB(dim) for _ in range(enc_blocks)])

        # A learnable mask token that will be used to fill in masked patch positions before decoding
        self.mask_token = nn.Parameter(torch.zeros(1, dim, 1, 1))

        # Decoder: a single ConvNeXt block (lightweight decoder as described in ConvNeXt V2)
        self.decoder_block = CNB(dim)

        # Output reconstruction: 
        # Use a 1x1 conv to map features to 3*(patch_size^2) channels, then PixelShuffle to reconstruct image.
        # PixelShuffle with upscale=4 will convert an (B, 3*16, 8, 8) feature map to (B, 3, 32, 32) output.
        self.conv_out = nn.Conv2d(dim, 3 * (4 * 4), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=4)

    def forward(self, x, mask):
        """
        x: input images tensor of shape (B, 3, H, W), expected 32x32 for CIFAR-10.
        mask: binary mask for patches of shape (B, 1, H_patch, W_patch), 
              where H_patch = H/4, W_patch = W/4 for patch size 4x4.
              mask == 1 indicates a masked (removed) patch, 0 indicates visible patch.
        """
        # Step 1: Patchify the input
        x = self.stem(x)  # shape: (B, dim, H_patch, W_patch)

        # Ensure mask is on the same device and correct shape
        mask = mask.to(x.device)
        # Step 2: Apply mask to patch embeddings (zero out masked patches to remove their information)
        x = x * (1 - mask)  # set masked patch embeddings to 0

        # Step 3: Encoder - process visible patches with ConvNeXt blocks
        for block in self.enc_blocks:
            x = block(x)
            # Re-apply mask after each block to prevent information leakage into masked positions
            x = x * (1 - mask)

        # Step 4: Prepare decoder input by inserting mask tokens at masked patch positions
        x = x + self.mask_token * mask  # replace masked areas with learned mask token embeddings

        # Step 5: Decoder - a ConvNeXt block processes the combined tokens (visible + mask token)
        x = self.decoder_block(x)

        # Step 6: Reconstruct the image from decoded features
        # Convert features to patch pixels
        x = self.conv_out(x)          # shape: (B, 3*16, H_patch, W_patch) i.e., (B, 48, 8, 8)
        x = self.pixel_shuffle(x)     # rearrange to (B, 3, H_patch*4, W_patch*4) = (B, 3, 32, 32)
        return x

# Utility function to generate a random mask for patches
def GRM(batch_size, patch_H, patch_W, mask_ratio, device):
    """
    Generate a random binary mask for patch positions.
    Returns a tensor of shape (batch_size, 1, patch_H, patch_W) 
    with `mask_ratio` portion of patches set to 1 (masked) and the rest 0.
    """
    num_patches = patch_H * patch_W
    # Number of patches to keep (not masked)
    num_keep = int(num_patches * (1 - mask_ratio))
    # Generate a random permutation of patch indices for each sample
    noise = torch.rand(batch_size, num_patches, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :num_keep]        # indices of patches to keep (visible)
    # Initialize mask with all 1s (mask everything), then mark kept indices as 0 (visible)
    mask_flat = torch.ones(batch_size, num_patches, device=device)
    mask_flat.scatter_(1, ids_keep, 0)          # set visible patch indices to 0
    mask = mask_flat.view(batch_size, 1, patch_H, patch_W)
    return mask

if __name__ == "__main__":
    # Hyperparameters for training
    num_epochs = 10
    batch_size = 64
    mask_ratio = 0.5  # 50% of patches will be masked
    learning_rate = 1e-3

    # Explanation (in comments):
    # Using 10 epochs and batch size 64 for training to ensure the demo runs on modest resources.
    # 10 epochs on CIFAR-10 is enough to get a basic learning signal without taking too long.
    # Batch size 64 is a reasonable trade-off between training speed and memory usage on a typical machine.

    # Load CIFAR-10 dataset
    transform = transforms.ToTensor()  # convert images to tensor (0-1 range)
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model, optimizer, and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNA(dim=32, enc_blocks=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            B, _, H, W = images.shape
            # Derive patch grid size from image (for CIFAR-10 32x32 -> 8x8 patches)
            patch_H, patch_W = H // 4, W // 4

            # Generate random mask for this batch
            mask = GRM(B, patch_H, patch_W, mask_ratio, device)

            # Forward pass
            outputs = model(images, mask)
            # Compute MSE loss between reconstructed and original images, only on masked patches
            # Upsample patch mask to pixel mask for the full image (1 for pixels that were masked)
            mask_pixel = F.interpolate(mask, scale_factor=4, mode='nearest')  # shape (B,1,H,W)
            # Calculate mean squared error on masked regions only
            loss = F.mse_loss(outputs * mask_pixel, images * mask_pixel)
            # (Alternatively: loss = ((outputs - images) ** 2 * mask_pixel).sum() / mask_pixel.sum())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)  # accumulate sum of loss for epoch (weighted by batch size)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Reconstruction Loss: {avg_loss:.4f}")

    # Switch model to evaluation mode for visualization
    model.eval()

    # Visualization: show original, masked, and reconstructed images for some test samples
    # Get a small batch of test images
    sample_batch = next(iter(torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True)))
    sample_images, sample_labels = sample_batch
    sample_images = sample_images.to(device)
    B, _, H, W = sample_images.shape
    patch_H, patch_W = H // 4, W // 4

    # Generate a random mask for visualization (can set a fixed seed for reproducibility if needed)
    torch.manual_seed(0)
    mask = GRM(B, patch_H, patch_W, mask_ratio, device)

    # Run the model to get reconstructed images
    with torch.no_grad():
        reconstructed = model(sample_images, mask)

    # Prepare images for plotting (move to CPU and convert to numpy)
    sample_images = sample_images.cpu()
    reconstructed = reconstructed.cpu()
    mask = mask.cpu()
    # Construct masked input images for display by zeroing out masked patches in the original
    mask_pixel = F.interpolate(mask, scale_factor=4, mode='nearest')  # upscale mask to full image size
    masked_input = sample_images * (1 - mask_pixel)  # set masked pixels to 0 (black)
    # Combine reconstructed patches with original visible parts for final reconstruction view
    recon_combined = sample_images * (1 - mask_pixel) + reconstructed * mask_pixel

    # Plot the results
    fig, axes = plt.subplots(nrows=B, ncols=3, figsize=(6, 2 * B))
    for i in range(B):
        # Original image
        axes[i, 0].imshow(sample_images[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        # Masked input image (what the encoder sees)
        axes[i, 1].imshow(masked_input[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Masked")
        axes[i, 1].axis('off')
        # Reconstructed image (with masked regions filled in)
        axes[i, 2].imshow(recon_combined[i].permute(1, 2, 0).numpy())
        axes[i, 2].set_title("Reconstructed")
        axes[i, 2].axis('off')
    plt.tight_layout()
    # Save the visualization to a file (helpful for non-interactive environments)
    plt.savefig("reconstruction_examples.png")
    plt.show()


def evaluate_top1(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            B, _, H, W = images.shape
            patch_H, patch_W = H // 4, W // 4
            mask = GRM(B, patch_H, patch_W, 0.5, device)  

            reconstructed = model(images, mask) 

            with torch.no_grad():
                feats = model.stem(images)
                for block in model.enc_blocks:
                    feats = block(feats)
                    feats = feats * (1 - mask)
                pooled = F.adaptive_avg_pool2d(feats, (1, 1)).view(B, -1)
                classifier = torch.nn.Linear(pooled.size(1), 10).to(device)
                preds = classifier(pooled).argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNA(dim=32, enc_blocks=2).to(device)
    transform = transforms.ToTensor()
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    top1_acc = evaluate_top1(model, test_loader, device)
    print(f"Top-1 Accuracy on CIFAR-10: {top1_acc * 100:.2f}%")