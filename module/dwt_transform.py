import torch
import torch.nn as nn

from pytorch_wavelets import DWTForward, DWTInverse


class Dwt2dEncoder(nn.Module):
    def __init__(self, wave, mode='zero'):
        super().__init__()
        self.xfm = DWTForward(J=2, wave=wave, mode=mode)

    def forward(self, x):
        Yl, Yh = self.xfm(x)
        dwt_4x2_images_from_4x4 = [torch.cat([Yl, Yh[1][:, :, 0]], dim=2),
                                   torch.cat([Yh[1][:, :, 1], Yh[1][:, :, 2]], dim=2)]
        dwt_2x2_image_from_4x2 = torch.cat(dwt_4x2_images_from_4x4, dim=3)
        dwt_2x2_images_from_4x2 = [dwt_2x2_image_from_4x2, Yh[0][:, :, 0], Yh[0][:, :, 1], Yh[0][:, :, 2]]
        dwt_2x2_features = torch.cat(dwt_2x2_images_from_4x2, dim=1)
        return dwt_2x2_features


class Dwt2dDecoder(nn.Module):
    def __init__(self, wave, mode='zero'):
        super().__init__()
        self.ifm = DWTInverse(wave=wave, mode=mode)

    def forward(self, dwt_2x2_features):
        b, c, w, h = dwt_2x2_features.shape
        dwt_2x2_image, Yh00, Yh01, Yh02 = torch.split(dwt_2x2_features, c // 4, dim=1)
        Yh0 = torch.stack([Yh00, Yh01, Yh02], dim=2)
        dwt_4x2_images_from_4x4 = torch.split(dwt_2x2_image, h // 2, dim=3)
        ((Yl, Yh10), (Yh11, Yh12)) = [torch.split(x, w // 2, dim=2) for x in dwt_4x2_images_from_4x4]
        Yh1 = torch.stack([Yh10, Yh11, Yh12], dim=2)
        Yh = [Yh0, Yh1]
        recon = self.ifm((Yl, Yh))
        return recon


if __name__ == '__main__':
    import cv2
    import numpy as np
    from lazuritetools.dl_utils.plot_utils import plt_show

    img = cv2.imread('Imgs/img.png')
    x = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0) / 255.0
    plt_show(x)
    print(x.shape)

    dwt_encoder = Dwt2dEncoder(wave='haar')
    dwt_decoder = Dwt2dDecoder(wave='haar')

    # x = torch.rand(10, 5, 128, 128)
    dwt_2x2_features = dwt_encoder(x)
    print(dwt_2x2_features.shape)

    recon = dwt_decoder(dwt_2x2_features)
    print("Recon Error:")
    print(torch.sum(torch.abs(recon - x)))

    plt_show(recon)