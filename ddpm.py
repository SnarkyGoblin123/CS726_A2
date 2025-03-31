import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# from sklearn.inspection import DecisionBoundaryDisplay

class NoiseScheduler:
    """
    Noise scheduler for the DDPM model.

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use ("linear" or "cosine")
        device: str, the device to use ("cuda" or "cpu")
        **kwargs: additional arguments for the scheduler
    """
    def __init__(self, num_timesteps=1000, type="linear", device="cuda", **kwargs):
        self.num_timesteps = num_timesteps
        self.type = type
        self.device = device

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == "cosine":
            self.init_cosine_schedule(**kwargs)
        elif type == "sigmoid":
            self.init_sigmoid_schedule(**kwargs)
        else:
            raise ValueError(f"Unknown schedule type: {type}")

        # Precompute useful quantities and move them to the specified device
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

    def init_linear_schedule(self, beta_start=1e-4, beta_end=0.02):
        """
        Initialize a linear variance schedule for beta.

        Args:
            beta_start: float, the starting value of beta
            beta_end: float, the ending value of beta
        """
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32, device=self.device)

    def init_cosine_schedule(self, s=0.008):
        """
        Initialize a cosine variance schedule for beta.

        Args:
            s: float, offset for the cosine schedule
        """
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps, dtype=torch.float32, device=self.device)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clip(betas, 0.0001, 0.9999)
    
    def init_sigmoid_schedule(self, beta_start=1e-4, beta_end=0.02, steepness=3.0):
        """
        Initialize a sigmoid variance schedule for beta.

        Args:
            beta_start: float, the starting value of beta
            beta_end: float, the ending value of beta
            steepness: float, controls the sharpness of the transition
        """
        t = torch.linspace(0, self.num_timesteps - 1, self.num_timesteps, dtype=torch.float32, device=self.device)
        midpoint = self.num_timesteps / 2
        self.betas = beta_start + (beta_end - beta_start) / (1 + torch.exp(-steepness * (t - midpoint) / midpoint))
        self.betas = torch.clip(self.betas, 0.0001, 0.9999)

    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to the data x_0 at timestep t.

        Args:
            x_0: torch.Tensor, the original data [batch_size, n_dim]
            t: torch.Tensor, the timestep for each sample in the batch [batch_size]
            noise: torch.Tensor, optional noise to add (if None, random noise is generated)

        Returns:
            torch.Tensor, the noisy data at timestep t [batch_size, n_dim]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get the noise schedule values for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]  # [batch_size]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]  # [batch_size]

        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)  # [batch_size, 1]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)  # [batch_size, 1]

        # Compute noisy data
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    def __len__(self):
        return self.num_timesteps

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for the timestep t.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings / torch.norm(embeddings, dim=-1, keepdim=True) 


class UNet1D(nn.Module):
    """
    1D U-Net-like architecture for noise prediction.
    """
    def __init__(self, n_dim=3, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Reduced Downsample blocks with activations
        self.down1 = nn.Sequential(
            nn.Linear(n_dim, 128),
            nn.SiLU()
        )
        self.norm1 = nn.LayerNorm(128)

        self.down2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU()
        )
        self.norm2 = nn.LayerNorm(256)

        # Bottleneck size reduced to 256
        self.bottleneck = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU()
        )
        self.norm_bottleneck = nn.LayerNorm(256)

        # Reduced Upsample blocks with activations
        self.up1 = nn.Sequential(
            nn.Linear(256 + 256, 256),  # Include skip connection
            nn.SiLU()
        )
        self.norm_up1 = nn.LayerNorm(256)

        # Reduce h's size from 256 to 128 for consistency
        self.reduce_proj = nn.Linear(256, 128)

        self.up2 = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.SiLU()
        )
        self.norm_up2 = nn.LayerNorm(128)

        self.up3 = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.SiLU()
        )
        self.norm_up3 = nn.LayerNorm(64)

        # Time conditioning layers (Projects time embedding to match feature sizes)
        self.time_proj_1 = nn.Linear(time_emb_dim, n_dim)
        self.time_proj_2 = nn.Linear(time_emb_dim, 128)
        self.time_proj_3 = nn.Linear(time_emb_dim, 256)
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, 256)  # Reduced to 256

        # Upsample time projections
        self.time_proj_up1 = nn.Linear(time_emb_dim, 256)
        self.time_proj_up2 = nn.Linear(time_emb_dim, 128)
        self.time_proj_up3 = nn.Linear(time_emb_dim, 128)

        # Final output layer
        self.final = nn.Linear(64, n_dim)

    def forward(self, x, t):
        """
        Forward pass for UNet1D.

        Args:
            x: Input data [batch_size, n_dim]
            t: Timesteps [batch_size] (integer timesteps)

        Returns:
            Predicted noise tensor [batch_size, n_dim]
        """
        # Compute time embedding
        t = self.time_mlp(t)  # [batch_size, time_emb_dim]

        # Downsample
        h1 = self.norm1(self.down1(x + self.time_proj_1(t)))  # [batch_size, 128]
        h2 = self.norm2(self.down2(h1 + self.time_proj_2(t)))  # [batch_size, 256]

        # Bottleneck
        h = self.norm_bottleneck(self.bottleneck(h2 + self.time_proj_bottleneck(t)))  # [batch_size, 256]

        # Upsample
        h = self.norm_up1(self.up1(torch.cat([h + self.time_proj_up1(t), h2], dim=-1)))  # [batch_size, 256]
        
        # Reduce h's size to 128 to match h1 size
        h = self.reduce_proj(h)  # [batch_size, 128]
        
        h = self.norm_up2(self.up2(torch.cat([h + self.time_proj_up2(t), h1], dim=-1)))  # [batch_size, 128]
        h = self.norm_up3(self.up3(torch.cat([h + self.time_proj_up3(t), h1], dim=-1)))  # [batch_size, 64]

        # Final output
        return self.final(h)


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.null_embedding = nn.Parameter(torch.zeros(1, embedding_dim))  # Learned null embedding

    def forward(self, x):
        # Replace -1 with the null embedding
        mask = (x == -1)
        x[mask] = 0  # Temporarily replace -1 with 0 to avoid IndexError
        embeddings = self.embedding(x)
        embeddings[mask] = self.null_embedding  # Replace with null embedding
        return embeddings

class ConditionalUNet1D(nn.Module):
    def __init__(self, n_dim=3, n_classes=2, time_emb_dim=512):
        """
        1D U-Net-like architecture for noise prediction with class conditioning.

        Args:
            n_dim: int, the dimensionality of the data
            n_classes: int, the number of classes
            time_emb_dim: int, the dimensionality of the time embeddings
        """
        super().__init__()
        self.class_embedding = ConditionalEmbedding(n_classes, time_emb_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Downsample blocks
        self.down1 = nn.Sequential(
            nn.Linear(n_dim, 128),
            nn.SiLU()
        )
        self.norm1 = nn.LayerNorm(128)

        self.down2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU()
        )
        self.norm2 = nn.LayerNorm(256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU()
        )
        self.norm_bottleneck = nn.LayerNorm(256)

        # Upsample blocks
        self.up1 = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.SiLU()
        )
        self.norm_up1 = nn.LayerNorm(256)

        self.reduce_proj = nn.Linear(256, 128)

        self.up2 = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.SiLU()
        )
        self.norm_up2 = nn.LayerNorm(128)

        self.up3 = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.SiLU()
        )
        self.norm_up3 = nn.LayerNorm(64)

        # Time and class conditioning layers
        self.time_proj_1 = nn.Linear(time_emb_dim, n_dim)
        self.time_proj_2 = nn.Linear(time_emb_dim, 128)
        self.time_proj_3 = nn.Linear(time_emb_dim, 256)
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, 256)
        self.time_proj_up1 = nn.Linear(time_emb_dim, 256)
        self.time_proj_up2 = nn.Linear(time_emb_dim, 128)
        self.time_proj_up3 = nn.Linear(time_emb_dim, 128)

        self.class_proj_1 = nn.Linear(time_emb_dim, n_dim)
        self.class_proj_2 = nn.Linear(time_emb_dim, 128)
        self.class_proj_3 = nn.Linear(time_emb_dim, 256)
        self.class_proj_bottleneck = nn.Linear(time_emb_dim, 256)
        self.class_proj_up1 = nn.Linear(time_emb_dim, 256)
        self.class_proj_up2 = nn.Linear(time_emb_dim, 128)
        self.class_proj_up3 = nn.Linear(time_emb_dim, 128)

        # Final output layer
        self.final = nn.Linear(64, n_dim)

    def forward(self, x, t, y):
        """
        Forward pass for ConditionalUNet1D.

        Args:
            x: Input data [batch_size, n_dim]
            t: Timesteps [batch_size]
            y: Class labels [batch_size]

        Returns:
            Predicted noise tensor [batch_size, n_dim]
        """
        # Compute time and class embeddings
        t_emb = self.time_mlp(t)  # [batch_size, time_emb_dim]
        y_emb = self.class_embedding(y)  # [batch_size, time_emb_dim]

        # Downsample
        h1 = self.norm1(self.down1(x + self.time_proj_1(t_emb) + self.class_proj_1(y_emb)))  # [batch_size, 128]
        h2 = self.norm2(self.down2(h1 + self.time_proj_2(t_emb) + self.class_proj_2(y_emb)))  # [batch_size, 256]

        # Bottleneck
        h = self.norm_bottleneck(self.bottleneck(h2 + self.time_proj_bottleneck(t_emb) + self.class_proj_bottleneck(y_emb)))  # [batch_size, 256]

        # Upsample
        h = self.norm_up1(self.up1(torch.cat([h + self.time_proj_up1(t_emb) + self.class_proj_up1(y_emb), h2], dim=-1)))  # [batch_size, 256]
        h = self.reduce_proj(h)  # [batch_size, 128]
        h = self.norm_up2(self.up2(torch.cat([h + self.time_proj_up2(t_emb) + self.class_proj_up2(y_emb), h1], dim=-1)))  # [batch_size, 128]
        h = self.norm_up3(self.up3(torch.cat([h + self.time_proj_up3(t_emb) + self.class_proj_up3(y_emb), h1], dim=-1)))  # [batch_size, 64]

        # Final output
        return self.final(h)

class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM.

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        """
        super(DDPM, self).__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps

        # U-Net-like architecture for noise prediction
        self.model = UNet1D(n_dim=n_dim)

    def forward(self, x, t):
        """
        Forward pass for the DDPM model.

        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        return self.model(x, t)

    
class ConditionalDDPM(nn.Module):
    def __init__(self, n_classes=2, n_dim=3, n_steps=200, time_emb_dim=512):
        """
        Class-dependent noise prediction network for the DDPM.

        Args:
            n_classes: int, number of classes in the dataset
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
            time_emb_dim: int, the dimensionality of the time embeddings
        """
        super(ConditionalDDPM, self).__init__()
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.n_steps = n_steps

        # U-Net-like architecture for noise prediction with class conditioning
        self.model = ConditionalUNet1D(n_dim=n_dim, n_classes=n_classes, time_emb_dim=time_emb_dim)

    def forward(self, x, t, y):
        """
        Forward pass for the ConditionalDDPM model.

        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]
            y: torch.Tensor, the class label tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        return self.model(x, t, y)

class ClassifierDDPM:
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.n_classes = model.n_classes

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        """
        Predict the class label for a given input using the trained ConditionalDDPM model.

        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]

        Returns:
            torch.Tensor, the predicted class tensor [batch_size]
        """
        self.model.eval()
        self.model.to(x.device)

        batch_size = x.shape[0]
        t = torch.zeros(batch_size, dtype=torch.long, device=x.device)  # Use t=0 for classification

        # Compute predicted noise for each class
        reconstruction_errors = []
        for class_label in range(self.n_classes):
            y = torch.full((batch_size,), class_label, device=x.device, dtype=torch.long)
            predicted_noise = self.model(x, t, y)

            # Reconstruct the input using the predicted noise
            alpha_t = self.noise_scheduler.alphas[0]
            alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[0]
            reconstructed_x = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

            # Compute reconstruction error (MSE)
            reconstruction_error = F.mse_loss(reconstructed_x, x, reduction='none').mean(dim=-1)
            reconstruction_errors.append(reconstruction_error)

        # Stack reconstruction errors and find the class with the minimum error
        reconstruction_errors = torch.stack(reconstruction_errors, dim=1)  # [batch_size, n_classes]
        predicted_classes = torch.argmin(reconstruction_errors, dim=1)  # [batch_size]

        return predicted_classes

    def predict_proba(self, x):
        """
        Predict the class probabilities for a given input using the trained ConditionalDDPM model.

        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]

        Returns:
            torch.Tensor, the predicted probabilities for each class [batch_size, n_classes]
        """
        self.model.eval()
        self.model.to(x.device)

        batch_size = x.shape[0]
        t = torch.zeros(batch_size, dtype=torch.long, device=x.device)  # Use t=0 for classification

        # Compute predicted noise for each class
        reconstruction_errors = []
        for class_label in range(self.n_classes):
            y = torch.full((batch_size,), class_label, device=x.device, dtype=torch.long)
            predicted_noise = self.model(x, t, y)

            # Reconstruct the input using the predicted noise
            alpha_t = self.noise_scheduler.alphas[0]
            alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[0]
            reconstructed_x = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

            # Compute reconstruction error (MSE)
            reconstruction_error = F.mse_loss(reconstructed_x, x, reduction='none').mean(dim=-1)
            reconstruction_errors.append(reconstruction_error)

        # Stack reconstruction errors and convert to probabilities
        reconstruction_errors = torch.stack(reconstruction_errors, dim=1)  # [batch_size, n_classes]
        probabilities = F.softmax(-reconstruction_errors, dim=1)  # [batch_size, n_classes]

        return probabilities

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the DDPM model.

    Args:
        model: DDPM, the noise prediction model
        noise_scheduler: NoiseScheduler, the noise scheduler
        dataloader: DataLoader, the dataloader for the dataset
        optimizer: Optimizer, the optimizer for training
        epochs: int, the number of epochs to train
        run_name: str, the name of the run for saving checkpoints
        device: str, the device to use ("cuda" or "cpu")
    """
    model.train()
    model.to(device)
    loss_per_epoch= []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_0, _ = batch  # Ignore labels if present
            else:
                x_0 = batch[0]  # Unlabeled dataset

            x_0 = x_0.to(device)
            t = torch.randint(0, noise_scheduler.num_timesteps, (x_0.shape[0],), device=device)  # Random timesteps
            noise = torch.randn_like(x_0)  # Random noise
            # print("x_0:", x_0.shape)
            # print(t.shape)
            # Add noise to the data
            x_t = noise_scheduler.add_noise(x_0, t, noise)
            # print(x_t.shape)
            # Predict the noise
            predicted_noise = model(x_t, t)
            loss = F.mse_loss(predicted_noise, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print epoch loss
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        loss_per_epoch.append(loss.item())
    # Save the model
    torch.save(model.state_dict(), f'{run_name}/model.pth')
    return loss_per_epoch


@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False):
    """
    Sample from the DDPM model.

    Args:
        model: DDPM, the noise prediction model
        noise_scheduler: NoiseScheduler, the noise scheduler
        n_samples: int, the number of samples to generate
        device: str, the device to use ("cuda" or "cpu")
        return_intermediate: bool, whether to return intermediate samples

    Returns:
        torch.Tensor, the generated samples [n_samples, n_dim]
    """
    model.eval()
    model.to(device)
    # print(model.n_dim)
    # Initialize from random noise (n-dimensional)
    x_t = torch.randn((n_samples, model.n_dim), device=device)  # Use model.n_dim instead of image_size

    if return_intermediate:
        intermediates = []

    # Reverse diffusion process
    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x_t, t_tensor)
        if torch.isnan(predicted_noise).any():
            print(f"Warning: NaN detected in predicted_noise at timestep {t}")

        # Compute reverse process mean
        alpha_t = noise_scheduler.alphas[t]
        alpha_cumprod_t = noise_scheduler.alphas_cumprod[t]
        beta_t = noise_scheduler.betas[t]

        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

        x_t = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        ) + torch.sqrt(beta_t) * noise

        if return_intermediate:
            intermediates.append(x_t.cpu())

    return torch.stack(intermediates) if return_intermediate else x_t

def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the ConditionalDDPM model.

    Args:
        model: ConditionalDDPM, the noise prediction model
        noise_scheduler: NoiseScheduler, the noise scheduler
        dataloader: DataLoader, the dataloader for the dataset
        optimizer: Optimizer, the optimizer for training
        epochs: int, the number of epochs to train
        run_name: str, the name of the run for saving checkpoints
    """
    model.train()
    model.to(device)
    loss_per_epoch = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_0, y = batch  # x_0: data, y: labels
            else:
                raise ValueError("Conditional training requires labeled data.")

            x_0 = x_0.to(device)
            y = y.to(device)
            t = torch.randint(0, noise_scheduler.num_timesteps, (x_0.shape[0],), device=device)
            noise = torch.randn_like(x_0)

            # Add noise to the data
            x_t = noise_scheduler.add_noise(x_0, t, noise)

            # Predict the noise
            predicted_noise = model(x_t, t, y)
            loss = F.mse_loss(predicted_noise, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
        loss_per_epoch.append(epoch_loss / len(dataloader))

    # Save the model
    torch.save(model.state_dict(), f'{run_name}/model.pth')
    return loss_per_epoch

@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, class_label):
    """
    Sample from the ConditionalDDPM model without Classifier-Free Guidance.

    Args:
        model: ConditionalDDPM, the noise prediction model
        n_samples: int, the number of samples to generate
        noise_scheduler: NoiseScheduler, the noise scheduler
        class_label: int, the class label for conditional generation

    Returns:
        torch.Tensor, the generated samples [n_samples, n_dim]
    """
    model.eval()
    model.to(device)

    # Initialize from random noise
    x_t = torch.randn((n_samples, model.n_dim), device=device)

    # Create a tensor for the class label
    y = torch.full((n_samples,), class_label, device=device, dtype=torch.long)

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # Predict noise with class conditioning
        predicted_noise = model(x_t, t_tensor, y)

        # Reverse diffusion process
        alpha_t = noise_scheduler.alphas[t]
        alpha_cumprod_t = noise_scheduler.alphas_cumprod[t]
        beta_t = noise_scheduler.betas[t]

        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

        x_t = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        ) + torch.sqrt(beta_t) * noise

    return x_t

def trainCFG(model, noise_scheduler, dataloader, optimizer, epochs, run_name, p_uncond=0.1):
    """
    Train the ConditionalDDPM model with Classifier-Free Guidance (CFG).

    Args:
        model: ConditionalDDPM, the noise prediction model
        noise_scheduler: NoiseScheduler, the noise scheduler
        dataloader: DataLoader, the dataloader for the dataset
        optimizer: Optimizer, the optimizer for training
        epochs: int, the number of epochs to train
        run_name: str, the name of the run for saving checkpoints
        p_uncond: float, the probability of dropping the conditioning (default: 0.1)
    """
    model.train()
    model.to(device)
    loss_per_epoch = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_0, y = batch  # x_0: data, y: labels
            else:
                raise ValueError("Conditional training requires labeled data.")

            x_0 = x_0.to(device)
            y = y.to(device)

            # Randomly drop the conditioning with probability p_uncond
            mask = torch.rand(y.shape[0], device=device) < p_uncond
            y[mask] = -1  # Use a special token (e.g., -1) for unconditional training

            t = torch.randint(0, noise_scheduler.num_timesteps, (x_0.shape[0],), device=device)
            noise = torch.randn_like(x_0)

            # Add noise to the data
            x_t = noise_scheduler.add_noise(x_0, t, noise)

            # Predict the noise
            predicted_noise = model(x_t, t, y)
            loss = F.mse_loss(predicted_noise, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
        loss_per_epoch.append(epoch_loss / len(dataloader))

    # Save the model
    torch.save(model.state_dict(), f'{run_name}/model.pth')
    return loss_per_epoch

@torch.no_grad()
def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the ConditionalDDPM model using Classifier-Free Guidance (CFG).

    Args:
        model: ConditionalDDPM, the noise prediction model
        n_samples: int, the number of samples to generate
        noise_scheduler: NoiseScheduler, the noise scheduler
        guidance_scale: float, the guidance strength (w)
        class_label: int, the class label for conditional generation

    Returns:
        torch.Tensor, the generated samples [n_samples, n_dim]
    """
    model.eval()
    model.to(device)

    # Initialize from random noise
    x_t = torch.randn((n_samples, model.n_dim), device=device)

    # Create a tensor for the class label
    y = torch.full((n_samples,), class_label, device=device, dtype=torch.long)

    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

        # Predict noise with and without conditioning
        predicted_noise_cond = model(x_t, t_tensor, y)
        predicted_noise_uncond = model(x_t, t_tensor, torch.full_like(y, -1))  # Unconditional prediction

        # Combine conditional and unconditional predictions using guidance scale
        predicted_noise = (1 + guidance_scale) * predicted_noise_cond - guidance_scale * predicted_noise_uncond

        # Reverse diffusion process
        alpha_t = noise_scheduler.alphas[t]
        alpha_cumprod_t = noise_scheduler.alphas_cumprod[t]
        beta_t = noise_scheduler.betas[t]

        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

        x_t = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        ) + torch.sqrt(beta_t) * noise

    return x_t


def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass


def train_mlp_classifier(real_data, real_labels):
    real_data = real_data.cpu().numpy()  # Move to CPU and convert to NumPy
    real_labels = real_labels.cpu().numpy()

    classifier = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=500)
    classifier.fit(real_data, real_labels)

    return classifier

def calculate_fid(real_data, generated_data):
    """
    Computes the FID score between real and generated samples.

    Args:
        real_data (torch.Tensor or np.ndarray): Real dataset samples [N, dim]
        generated_data (torch.Tensor or np.ndarray): Generated samples [N, dim]

    Returns:
        float: FID score
    """
    # Convert to NumPy if necessary
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.cpu().numpy()
    if isinstance(generated_data, torch.Tensor):
        generated_data = generated_data.cpu().numpy()

    # Compute mean and covariance
    mu_real, sigma_real = np.mean(real_data, axis=0), np.cov(real_data, rowvar=False)
    mu_gen, sigma_gen = np.mean(generated_data, axis=0), np.cov(generated_data, rowvar=False)

    # Compute squared difference of means
    mean_diff = np.sum((mu_real - mu_gen) ** 2)

    # Compute square root of product of covariance matrices
    cov_sqrt = sqrtm(sigma_real @ sigma_gen)
    
    # Numerical stability check
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid_score = mean_diff + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
    return fid_score



def evaluate_guidance_scale(model, noise_scheduler, classifier, real_data, real_labels, guidance_scales, n_classes, num_samples, run_name, output_dir="results"):
    """
    Evaluate different guidance scales by computing classification accuracy, FID, 
    and generating a single decision boundary plot for all classes.

    Args:
        model (ConditionalDDPM): The generative model.
        noise_scheduler (NoiseScheduler): The noise scheduler.
        classifier: Pre-trained classifier for accuracy evaluation.
        real_data (torch.Tensor): Real dataset samples for FID.
        real_labels (torch.Tensor): Real dataset labels.
        guidance_scales (list): List of guidance scale values.
        num_samples (int): Number of generated samples.
        output_dir (str): Directory to save results.

    Returns:
        None (saves results and plots in files)
    """
    # Ensure output directory existsrun_dir = os.path.join(output_dir, run_name)  # Create directory with run name
    run_dir = os.path.join(str(run_name), output_dir)  # Create directory with run name
    os.makedirs(run_dir, exist_ok=True)  # Ensure directory exists
    results_file = os.path.join(run_dir, "guidance_evaluation.txt")

    real_data = real_data.cpu().numpy()
    real_labels = real_labels.cpu().numpy()
    num_classes = n_classes
    print(run_name)
    model.load_state_dict(torch.load(f'{run_name}/model.pth'))

    with open(results_file, "w") as f:
        f.write("Guidance Scale Evaluation Results\n")
        f.write("=" * 50 + "\n")

        for scale in guidance_scales:
            accuracies = []
            fid_scores = []
            
            # Collect generated samples and labels for all classes
            generated_samples = []
            generated_labels = []

            for class_label in range(num_classes):
                samples = sampleCFG(
                    model, n_samples=num_samples // num_classes, noise_scheduler=noise_scheduler, 
                    guidance_scale=scale, class_label=class_label
                ).cpu().numpy()

                generated_samples.append(samples)
                generated_labels.append(np.full(len(samples), class_label))

            generated_samples = np.vstack(generated_samples)
            generated_labels = np.concatenate(generated_labels)

            # Compute classification accuracy for all classes
            predicted_labels = classifier.predict(generated_samples)
            avg_accuracy = accuracy_score(generated_labels, predicted_labels)
            
            # Compute FID for all classes
            fid_score = calculate_fid(real_data, generated_samples)

            # Save results to text file
            f.write(f"Guidance Scale {scale}:\n")
            f.write(f"  Avg Accuracy = {avg_accuracy:.4f}\n")
            f.write(f"  Avg FID = {fid_score:.4f}\n")
            f.write("-" * 50 + "\n")

            print(f"Guidance Scale {scale}: Avg Accuracy = {avg_accuracy:.4f}, Avg FID = {fid_score:.4f}")

            # Fit scaler for visualization
            scaler = StandardScaler()
            all_data = np.vstack([real_data, generated_samples])
            all_data_scaled = scaler.fit_transform(all_data)

            real_scaled = scaler.transform(real_data)
            gen_scaled = scaler.transform(generated_samples)

            # Create mesh grid for decision boundaries
            xx, yy = np.meshgrid(
                np.linspace(all_data_scaled[:, 0].min(), all_data_scaled[:, 0].max(), 100),
                np.linspace(all_data_scaled[:, 1].min(), all_data_scaled[:, 1].max(), 100)
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = classifier.predict(grid).reshape(xx.shape)

            # Plot everything in one figure
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
            
            # Plot real and generated data for each class
            for class_label in range(num_classes):
                plt.scatter(real_scaled[real_labels == class_label, 0], real_scaled[real_labels == class_label, 1], 
                            label=f"Real Class {class_label}", alpha=0.5, edgecolors='k')
                plt.scatter(gen_scaled[generated_labels == class_label, 0], gen_scaled[generated_labels == class_label, 1], 
                            label=f"Generated Class {class_label} (Scale {scale})", alpha=0.5, marker="x")

            plt.xlabel("Feature 1 (Scaled)")
            plt.ylabel("Feature 2 (Scaled)")
            plt.title(f"Decision Boundary & Data for Guidance Scale {scale}")
            plt.legend()

            # Save plot
            plot_path = os.path.join(run_dir, f"guidance_scale_{scale}.png")
            plt.savefig(plot_path)
            plt.close()
        

if __name__ == "__main__":
    losses = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample', '1.2.2', '1.2.3'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default = None)
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--n_dim", type=int, default = None)
    parser.add_argument("--n_classes", type=int, default = 2)
    parser.add_argument("--model", type=str, default="DDPM")

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_name = f'exps_unet/{args.model}_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}_{args.batch_size}_{args.epochs}' # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)
    if args.model == "DDPM":
        model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    elif args.model == "conditionalDDPM":
        model = ConditionalDDPM(n_dim=args.n_dim, n_steps=args.n_steps, n_classes=args.n_classes)
    elif args.model == "CFG":
        model = ConditionalDDPM(n_dim=args.n_dim, n_steps=args.n_steps, n_classes=args.n_classes)
    else:
        raise ValueError(f"Invalid model {args.model}")

    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta, device=device, type="linear")
    # noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, type="cosine")
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        dataset_output = dataset.load_dataset(args.dataset)

        # Check if dataset_output has labels or not
        if isinstance(dataset_output, tuple) and len(dataset_output) == 2:
            data_X, data_y = dataset_output  # Dataset has labels
            if data_y is None:  
                dataset_with_labels = False  # Handle case where labels are None
            else:
                data_y = data_y.to(device)
                dataset_with_labels = True
        else:
            data_X = dataset_output  # Dataset has only features
            dataset_with_labels = False

        # Move data to the correct device
        data_X = data_X.to(device)

        # Create DataLoader based on dataset type
        if dataset_with_labels:
            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(data_X, data_y),
                batch_size=args.batch_size,
                shuffle=True
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(data_X),  # No labels
                batch_size=args.batch_size,
                shuffle=True
            )

        if args.model == "DDPM":
            losses.append(train(model, noise_scheduler, dataloader, optimizer, epochs, run_name))
        elif args.model == "CFG":
            losses.append(trainCFG(model, noise_scheduler, dataloader, optimizer, epochs, run_name))
        elif args.model == "conditionalDDPM":
            losses.append(trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name))
        else:
            raise ValueError(f"Invalid model {args.model}")

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))

        if args.model == "DDPM":
            samples = sample(model, args.n_samples, noise_scheduler)
        elif args.model == "CFG":
            samples = sampleCFG(model, args.n_samples, noise_scheduler, guidance_scale=0, class_label=1)
        elif args.model == "conditionalDDPM":
            samples = sampleConditional(model, args.n_samples, noise_scheduler, class_label=1)
        else:
            raise ValueError(f"Invalid model {args.model}")
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
    elif args.mode == '1.2.2':
        # Load dataset
        real_data, real_labels = dataset.load_dataset(args.dataset)
        real_data, real_labels = real_data.to(device), real_labels.to(device)

        # Train the MLP classifier
        classifier = train_mlp_classifier(real_data, real_labels)
        print(run_name)
        # Test different guidance scales
        guidance_scales = [0, 1, 3, 5, 8]

        evaluate_guidance_scale(model, noise_scheduler, classifier, real_data, real_labels, guidance_scales, args.n_classes, args.n_samples, run_name)
    
    elif args.mode == '1.2.3':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        model.to(device)
        model.eval()

        # Initialize ClassifierDDPM
        classifier_ddpm = ClassifierDDPM(model, noise_scheduler)

        # Load real dataset
        real_data, real_labels = dataset.load_dataset(args.dataset)
        real_data, real_labels = real_data.to(device), real_labels.to(device)

        # Train MLP Classifier (Mode 1.2.2)
        mlp_classifier = train_mlp_classifier(real_data, real_labels)
        pred_labels_ddpm = classifier_ddpm.predict(real_data)
        accuracy_ddpm = accuracy_score(real_labels.cpu().numpy(), pred_labels_ddpm.cpu().numpy())

        # Predictions using MLP Classifier (Mode 1.2.2)
        pred_labels_mlp = mlp_classifier.predict(real_data.cpu().numpy())
        accuracy_mlp = accuracy_score(real_labels.cpu().numpy(), pred_labels_mlp)

        # Compare Results
        print(f"ClassifierDDPM Accuracy: {accuracy_ddpm:.4f}")
        print(f"MLP Classifier Accuracy (Mode 1.2.2): {accuracy_mlp:.4f}")
    else:
        raise ValueError(f"Invalid mode {args.mode}")


final_loss = [loss[-1] for loss in losses]
print("Final Loss: ",final_loss)


