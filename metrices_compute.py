import numpy as np
import cv2
import glob
from skimage import metrics
import torch
import torch.hub
from lpips.lpips import LPIPS
import os
import imageio

photometric = {
    "mse": None,
    "ssim": None,
    "psnr": None,
    "lpips": None
}

def compute_img_metric(im1t: torch.Tensor, im2t: torch.Tensor,
                       metric="mse", margin=0, mask=None):
    """
    im1t, im2t: torch.tensors with batched imaged shape, range from (0, 1)
    """
    if metric not in photometric.keys():
        raise RuntimeError(f"img_utils:: metric {metric} not recognized")
    if photometric[metric] is None:
        if metric == "mse":
            photometric[metric] = metrics.mean_squared_error
        elif metric == "ssim":
            photometric[metric] = metrics.structural_similarity
        elif metric == "psnr":
            photometric[metric] = metrics.peak_signal_noise_ratio
        elif metric == "lpips":
            photometric[metric] = LPIPS().cpu()

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1:
            mask = mask.expand(-1, 3, -1, -1)
        mask = mask.permute(0, 2, 3, 1).numpy()
        batchsz, hei, wid, _ = mask.shape
        if margin > 0:
            marginh = int(hei * margin) + 1
            marginw = int(wid * margin) + 1
            mask = mask[:, marginh:hei - marginh, marginw:wid - marginw]

    # convert from [0, 1] to [-1, 1]
    im1t = (im1t * 2 - 1).clamp(-1, 1)
    im2t = (im2t * 2 - 1).clamp(-1, 1)

    if im1t.dim() == 3:
        im1t = im1t.unsqueeze(0)
        im2t = im2t.unsqueeze(0)
    im1t = im1t.detach().cpu()
    im2t = im2t.detach().cpu()

    if im1t.shape[-1] == 3:
        im1t = im1t.permute(0, 3, 1, 2)
        im2t = im2t.permute(0, 3, 1, 2)

    im1 = im1t.permute(0, 2, 3, 1).numpy()
    im2 = im2t.permute(0, 2, 3, 1).numpy()
    batchsz, hei, wid, _ = im1.shape
    if margin > 0:
        marginh = int(hei * margin) + 1
        marginw = int(wid * margin) + 1
        im1 = im1[:, marginh:hei - marginh, marginw:wid - marginw]
        im2 = im2[:, marginh:hei - marginh, marginw:wid - marginw]
    values = []

    for i in range(batchsz):
        if metric in ["mse", "psnr"]:
            if mask is not None:
                im1 = im1 * mask[i]
                im2 = im2 * mask[i]
            value = photometric[metric](
                im1[i], im2[i]
            )
            if mask is not None:
                hei, wid, _ = im1[i].shape
                pixelnum = mask[i, ..., 0].sum()
                value = value - 10 * np.log10(hei * wid / pixelnum)
        elif metric in ["ssim"]:
            value, ssimmap = photometric["ssim"](
                im1[i], im2[i], multichannel=True, full=True
            )
            if mask is not None:
                value = (ssimmap * mask[i]).sum() / mask[i].sum()
        elif metric in ["lpips"]:
            value = photometric[metric](
                im1t[i:i + 1], im2t[i:i + 1]
            )
        else:
            raise NotImplementedError
        values.append(value)

    return sum(values) / len(values)


def main():
    # render_dir = "F:/deblur/AnnaTrain/images/*.png"
    # image_paths = glob.glob(render_dir)
    # img_idxs = []
    # for i in range(0,len(image_paths),6):
    #     path = image_paths[i]
    #     img_idxs.append(int(path.split("\\")[-1][:-4]))
    img_idxs = [  0,   6,  12,  18,  24,  30,  36,  42,  48,  54,  60,  66,  72,  78,  84,  90,  96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210,
                216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318,
                324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384, 390, 396, 402, 408, 414, 420, 426,
                432, 438, 444, 450, 456, 462, 468, 474, 480, 486, 492, 498, 504, 510, 516]

    render_imgs = []
    dsnerf_paths = "F:/deblur/AnnaTrain/dsnerf/room/*.png"
    dsnerf_paths = glob.glob(dsnerf_paths)
    for i in range(len(dsnerf_paths)):
        path = dsnerf_paths[i]
        if "depth" in path: continue
        render_imgs.append(imageio.imread(path)[..., :3] / 255.)
    render_imgs = np.array(render_imgs)
    img_idxs = np.array(img_idxs)

    
    dataset_dir = "F:/deblur/AnnaTrain/images/*.png"
    target_image_paths = np.array(glob.glob(dataset_dir))[img_idxs]
    target_imgs = np.array([imageio.imread(f)[..., :3] / 255. for f in target_image_paths])

    # evaluation
    rgbs = torch.tensor(render_imgs).float()
    target_rgb_ldr = torch.tensor(target_imgs).float()
    test_mse = compute_img_metric(rgbs, target_rgb_ldr, 'mse')
    test_psnr = compute_img_metric(rgbs, target_rgb_ldr, 'psnr')
    test_ssim = compute_img_metric(rgbs, target_rgb_ldr, 'ssim')
    test_lpips = compute_img_metric(rgbs, target_rgb_ldr, 'lpips')
    if isinstance(test_lpips, torch.Tensor):
        test_lpips = test_lpips.item()

    print(f"MSE:{test_mse:.8f} PSNR:{test_psnr:.8f} SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}\n")

if __name__ == "__main__":
    main()