import matplotlib.pyplot as plt
import torch

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import cosine_beta_schedule, sigmoid_beta_schedule
from denoising_diffusion_pytorch.simple_diffusion import logsnr_schedule_cosine, logsnr_schedule_shifted


def get_sqrt_alpha_cumprod_schedule(schedule_fn, *args, **kwargs):
    beta_schedule = schedule_fn(*args, **kwargs)
    alpha_schedule = torch.sqrt(1.0 - beta_schedule)
    alpha_cumprod_schedule = torch.cumprod(alpha_schedule, dim=0)
    sqrt_alpha_cumprod_schedule = torch.sqrt(alpha_cumprod_schedule)
    return sqrt_alpha_cumprod_schedule


def log_snr_schedule_to_full_schedule(log_snr_fn, timesteps, *args, **kwargs):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    log_snr = torch.tensor([log_snr_fn(i) for i in t], dtype=torch.float64)
    alpha_cumprod_schedule = torch.sqrt(log_snr.sigmoid())
    return alpha_cumprod_schedule


plt.plot(get_sqrt_alpha_cumprod_schedule(cosine_beta_schedule,
                                         timesteps=1000),
         label='cosine alpha schedule')
plt.plot(get_sqrt_alpha_cumprod_schedule(sigmoid_beta_schedule,
                                         timesteps=1000),
         label='sigmoid alpha schedule')

plt.plot(log_snr_schedule_to_full_schedule(logsnr_schedule_cosine,
                                           timesteps=1000),
         label='logsnr schedule')

plt.plot(log_snr_schedule_to_full_schedule(logsnr_schedule_shifted(logsnr_schedule_cosine, image_d=128, noise_d=32),
                                           timesteps=1000),
         label='logsnr schedule shifted 32')

plt.plot(log_snr_schedule_to_full_schedule(logsnr_schedule_shifted(logsnr_schedule_cosine, image_d=128, noise_d=64),
                                           timesteps=1000),
         label='logsnr schedule shifted 64')
plt.legend()
plt.title('Beta')
plt.show()
