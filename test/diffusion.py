import torch
from lazuritetools.dl_utils.import_utils import instantiate_from_config
from omegaconf import OmegaConf

config = OmegaConf.load('config/simple_diffusion.yaml')
model = instantiate_from_config(config.model)
diffusion = instantiate_from_config(config.diffusion,
                                    model=model)

# input = torch.randn(1, 3, 128, 128)
output = diffusion.sample(16)
print(output.shape)