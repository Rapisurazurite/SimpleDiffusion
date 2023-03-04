import sys

sys.path.insert(0, 'pytorch-wavelets')

import torch
from omegaconf import OmegaConf

from pytorch_wavelets.dwt.transform2d import DWTForward, DWTInverse
from lazuritetools.dl_utils.import_utils import instantiate_from_config
from lazuritetools.dl_utils.collate_utils import collate
from lazuritetools.dl_utils.plot_utils import plt_show

#%%
dataset_config = OmegaConf.load('config/dataset_128x128.yaml')
dataset = instantiate_from_config(dataset_config.train_dataset)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=10,
                                         collate_fn=collate,
                                         shuffle=False)
data_iter = iter(dataloader)
image, gt = next(data_iter)

#%%
# DWT Image
xfm = DWTForward(J=2, wave='haar', mode='zero')
Yl, Yh = xfm(image)
print(image.shape)
print(Yl.shape)
print(Yh[0].shape)
print(Yh[1].shape)

# stack to a larger image
dwt_4x2_images_from_4x4 = [torch.cat([Yl, Yh[1][:, 0]], dim=2), torch.cat([Yh[1][:, 1], Yh[1][:, 2]], dim=2)]
dwt_2x2_image_from_4x2 = torch.cat(dwt_4x2_images_from_4x4, dim=3)
plt_show(dwt_2x2_image_from_4x2[0], 'dwt_2x2_image_from_4x2')

dwt_2x2_images_from_4x2 = [dwt_2x2_image_from_4x2, Yh[0][:, 0], Yh[0][:, 0], Yh[0][:, 2]]
dwt_1x2_images_from_2x2 = [torch.cat([dwt_2x2_images_from_4x2[0], dwt_2x2_images_from_4x2[1]], dim=2),
                           torch.cat([dwt_2x2_images_from_4x2[2], dwt_2x2_images_from_4x2[3]], dim=2)]
dwt_1x1_image_from_1x2 = torch.cat(dwt_1x2_images_from_2x2, dim=3)
plt_show(dwt_1x1_image_from_1x2[0], 'dwt_1x1_image_from_1x2')
dwt_2x2_features = torch.cat(dwt_2x2_images_from_4x2, dim=1)

ifm = DWTInverse(wave='haar', mode='zero')
Y = ifm((Yl, Yh))
plt_show(image[0], 'image')
plt_show(Y[0], 'Y')

assert torch.sum(torch.abs(Y - image)) < 0.1