import torch
import numpy as np

from joel_unet import DDPM
from joel_unet import NoiseScheduler

device = 'cuda:1'
prior_samples_path = "data/albatross_prior_samples.npy"
output_file = "albatross_samples.npy"

model = DDPM(n_dim=64, n_steps=100).to(device)
model.load_state_dict(torch.load("exps_unet/DDPM_64_100_0.0001_0.05_albatross_128_100/model.pth"))
print("model loaded")
model.eval()

noise_scheduler = NoiseScheduler(num_timesteps=100, beta_start=1e-4, beta_end=0.05, device=device, type="linear")

# Load prior samples (instead of random noise)
xT = torch.tensor(np.load(prior_samples_path), dtype=torch.float32, device=device)
final_samples = torch.zeros_like(xT,device=device)
num_samples, _ = xT.shape
BATCH_SIZE = 128

with torch.no_grad():
    for start in range(0, num_samples, BATCH_SIZE):
        print(start)
        end = min(start + BATCH_SIZE, num_samples)
        x_t = xT[start:end].clone().to(device)  # Load batch to GPU

        for t in reversed(range(noise_scheduler.num_timesteps)):
            t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.long, device=device)
            predicted_noise = model(x_t, t_tensor)  

            alpha_t = noise_scheduler.alphas[t]
            alpha_cumprod_t = noise_scheduler.alphas_cumprod[t]

            # Deterministic reverse process (z = 0, so no noise added)
            x_t = (1 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            )

        final_samples[start:end] = x_t 
        del x_t, predicted_noise, t_tensor
        torch.cuda.empty_cache()

# Save final output
np.save(output_file, final_samples.cpu().numpy())
print(f"Reproduced samples saved to {output_file}")


