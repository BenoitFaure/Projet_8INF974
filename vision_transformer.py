import torch
from torch import nn

# Helper function
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Transpose layer
class TransposeLayer(nn.Module):
    def __init__(self, dim1, dim2):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return torch.transpose(x, self.dim1, self.dim2)

# Class defining a Vision Transformer as described in the paper
#   - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale : https://arxiv.org/abs/2010.11929 (Note: copilot gave the papers link...)
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()

        # Get image size and patch size
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # Get patch dim
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'image dimensions must be divisible by the patch size'
    
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width


        # Patch embedding - Split image into patches and then linearly embed them
        self.patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size, padding=0),
            TransposeLayer(1, 2),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )


        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Extra learnable class token
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=0.1
        ), num_layers=depth)

        # Classifier head
        self.head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # Get batch size
        b = img.shape[0]

        # Patch embedding
        x = self.patch_embedding(img)#.transpose(1, 2)

        # Class token
        class_token = self.class_token.expand(b, -1, -1)
        x = torch.cat((class_token, x), dim=1)

        # Add position embedding
        x = x + self.pos_embedding

        # Transformer
        x = self.transformer(x)

        # Pool method (None for now)
        x = x[:, 0]

        # Classifier head
        x = self.head(x)

        return x