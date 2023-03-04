import torch
from lazuritetools.dl_utils.import_utils import instantiate_from_config
from lazuritetools.dl_utils.model_utils import count_params
from omegaconf import OmegaConf

config = OmegaConf.load('config/simple_diffusion.yaml')
model = instantiate_from_config(config.model)
count_params(model, verbose=True)


input = torch.randn(1, 3, 128, 128)
time = torch.zeros([1]).float().uniform_(0, 1)
output = model(input, time)
print(output.shape)