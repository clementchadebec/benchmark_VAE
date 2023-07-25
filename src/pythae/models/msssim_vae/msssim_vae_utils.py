# Code coming from https://github.com/jorge-pessoa/pytorch-msssim/blob/dev/pytorch_msssim/

import numpy as np
import torch
import torch.nn.functional as F


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(MSSSIM, self).__init__()
        self.window_size = window_size

    def _gaussian(self, sigma):
        gauss = torch.Tensor(
            [
                np.exp(-((x - self.window_size // 2) ** 2) / float(2 * sigma ** 2))
                for x in range(self.window_size)
            ]
        )
        return gauss / gauss.sum()

    def _create_window(self, n_dim, channel=1):
        _1D_window = self._gaussian(1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float()

        _3D_window = (
            torch.stack([_2D_window * x for x in _1D_window], dim=2)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)

        if n_dim == 3:
            return _3D_window.expand(
                channel, 1, self.window_size, self.window_size, self.window_size
            ).contiguous()
        else:
            return _2D_window.expand(
                channel, 1, self.window_size, self.window_size
            ).contiguous()

    def ssim(self, img1: torch.Tensor, img2: torch.Tensor):
        n_dim = len(img1.shape) - 2
        padd = int(self.window_size / 2)

        if n_dim == 2:
            (_, channel, height, width) = img1.size()
            convFunction = F.conv2d
        elif n_dim == 3:
            (_, channel, height, width, depth) = img1.size()
            convFunction = F.conv3d

        window = self._create_window(n_dim=n_dim, channel=channel).to(img1.device)

        mu1 = convFunction(img1, window, padding=padd, groups=channel)
        mu2 = convFunction(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            convFunction(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        )
        sigma2_sq = (
            convFunction(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        )
        sigma12 = (
            convFunction(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
        )

        L = 1  # data already in [0, 1]
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        ret = ssim_map.mean()
        return ret, cs

    def forward(self, img1, img2):
        n_dim = len(img1.shape) - 2
        if min(img1.shape[-n_dim:]) < 4:
            weights = torch.FloatTensor([1.0]).to(img1.device)

        elif min(img1.shape[-n_dim:]) < 8:
            weights = torch.FloatTensor([0.3222, 0.6778]).to(img1.device)

        elif min(img1.shape[-n_dim:]) < 16:
            weights = torch.FloatTensor([0.4558, 0.1633, 0.3809]).to(img1.device)

        elif min(img1.shape[-n_dim:]) < 32:
            weights = torch.FloatTensor([0.3117, 0.3384, 0.2675, 0.0824]).to(
                img1.device
            )

        else:
            weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(
                img1.device
            )
        levels = weights.size()[0]
        mssim = []
        mcs = []

        pool_size = [2] * n_dim
        if n_dim == 2:
            pool_function = F.avg_pool2d
        elif n_dim == 3:
            pool_function = F.avg_pool3d

        for _ in range(levels):
            sim, cs = self.ssim(img1, img2)
            mssim.append(sim)
            mcs.append(cs)

            img1 = pool_function(img1, pool_size)
            img2 = pool_function(img2, pool_size)

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # Normalize (to avoid NaNs during training unstable models, not compliant with original
        # definition)
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights

        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1 - output
