import torch
from lazuritetools.dl_utils.import_utils import instantiate_from_config
from lazuritetools.dl_utils.model_utils import count_params
from omegaconf import OmegaConf

config = OmegaConf.load('config/simple_diffusion_dwt.yaml')
dwt2d_encoder, dwt2d_decoder = instantiate_from_config(config.init_img_transform), instantiate_from_config(config.final_img_itransform)

model = instantiate_from_config(config.model,
                                init_img_transform=dwt2d_encoder,
                                final_img_itransform=dwt2d_decoder).cuda()
count_params(model, verbose=True)


input = torch.randn(1, 3, 256, 256).cuda()
time = torch.zeros([1]).float().uniform_(0, 1).cuda()
output = model(input, time)
print(output.device)
print(output.shape)