import copy 
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class EMA:
    def __init__(self, model, beta=0.999):
        """
        Initialize EMA with a model.
        
        Args:
            model: Model to create EMA of
            beta: Decay rate for EMA
        """
        self.beta = beta
        self.shadow = {}
        self.backup = {}
        
        # Create a deep copy for EMA
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        # Register shadow parameters
        self.register(model)

    def register(self, model):
        """Register model parameters for EMA tracking."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def step_ema(self, ema_model, model):
        """Update EMA parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data = self.beta * ema_param.data + (1 - self.beta) * model_param.data
            
            # Also update shadow parameters
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = self.beta * self.shadow[name] + (1 - self.beta) * param.data

    def get_ema_model(self):
        """Get the EMA model."""
        return self.ema_model

    def state_dict(self):
        """Returns both shadow parameters and EMA model state."""
        return {
            'shadow': {k: v.cpu() for k, v in self.shadow.items()},
            'ema_model': self.ema_model.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Loads both shadow parameters and EMA model state."""
        if 'shadow' in state_dict:
            for k, v in state_dict['shadow'].items():
                if k in self.shadow:
                    self.shadow[k] = v.clone()
                else:
                    raise KeyError(f"Unexpected key '{k}' in EMA shadow state")
        
        if 'ema_model' in state_dict:
            self.ema_model.load_state_dict(state_dict['ema_model'])

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256): # emb_dim = 256
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256): #emb_dim=256
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# NOTE: DO NOT USE 
class UNet_conditional(nn.Module):
    def __init__(self, c_in=5, c_out=5, time_dim=256, num_classes=None, device="cuda"): # c_in=5, c_out=5
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


####################################################################################################
# NOTE: USE THIS MODEL
class UNet_conditional_conv(nn.Module):
    def __init__(
        self, 
        c_in=5, 
        c_out=5, 
        time_dim=256, 
        y_dim=1
    ):
        """
        Initializes the conditional U-Net model.
        
        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            time_dim (int): Dimension of the time encoding.
            y_dim (int): Dimension of the label input.
        """
        super(UNet_conditional_conv, self).__init__()
        self.time_dim = time_dim
        self.y_dim = y_dim

        # Initial convolution layers
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)

        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # Up-convolution layers
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        # Label embedding for continuous labels
        if y_dim is not None:
            self.label_emb = nn.Linear(y_dim, time_dim)
        else:
            self.label_emb = None

    def pos_encoding(self, t, channels):
        """
        Generates positional encoding for the time steps.
        
        Args:
            t (Tensor): Time step tensor.
            channels (int): Number of channels for encoding.
        
        Returns:
            Tensor: Positional encoded tensor.
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input image tensor.
            t (Tensor): Time step tensor.
            y (Tensor): Label tensor (e.g., redshift values).
        
        Returns:
            Tensor: Output tensor.
        """
        # Positional encoding for time steps
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Incorporate label information if available
        if y is not None and self.label_emb is not None:
            y_emb = self.label_emb(y)
            t = t + y_emb  # Integrate label embedding with time encoding

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        # Bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Decoder
        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        return output


import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_lr = [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
            return cosine_lr