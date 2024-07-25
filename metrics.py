import os
import argparse
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import re

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def resize_image(image, scale):
    width, height = image.size
    return image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

def calculate_ssim(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    img1_gray = np.dot(img1[..., :3], [0.2989, 0.587, 0.114])
    img2_gray = np.dot(img2[..., :3], [0.2989, 0.587, 0.114])
    return ssim(img1_gray, img2_gray, data_range=img2_gray.max() - img2_gray.min())

def calculate_psnr(img1, img2):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_lpips(img1, img2, lpips_model):
    img1 = lpips.im2tensor(np.array(img1))
    img2 = lpips.im2tensor(np.array(img2))
    return lpips_model(img1, img2).item()

def main(args):
    test_folder = args.test_folder
    original_folder = args.original_folder
    scale = args.scale

    lpips_model = lpips.LPIPS(net='alex').to(torch.device('cpu'))

    ssim_scores = []
    psnr_scores = []
    lpips_scores = []

    test_filename_pattern = re.compile(r'frame_(\d{3})_camera_01\.png')

    for filename in os.listdir(test_folder):
        match = test_filename_pattern.match(filename)
        if match:
            test_image_path = os.path.join(test_folder, filename)
            num = match.group(1)
            original_filename = f"000{num}.png"
            original_image_path = os.path.join(original_folder, original_filename)

            if os.path.exists(original_image_path):
                test_image = load_image(test_image_path)
                original_image = load_image(original_image_path)
                original_image = resize_image(original_image, scale)

                ssim_score = calculate_ssim(test_image, original_image)
                psnr_score = calculate_psnr(test_image, original_image)
                lpips_score = calculate_lpips(test_image, original_image, lpips_model)

                ssim_scores.append(ssim_score)
                psnr_scores.append(psnr_score)
                lpips_scores.append(lpips_score)
            else:
                print(f"Original image for {filename} ({original_filename}) not found.")
        else:
            print(f"Filename {filename} does not match the expected pattern.")

    if ssim_scores and psnr_scores and lpips_scores:
        average_ssim = np.mean(ssim_scores)
        average_psnr = np.mean(psnr_scores)
        average_lpips = np.mean(lpips_scores)

        print(f"Average SSIM: {average_ssim}")
        print(f"Average PSNR: {average_psnr}")
        print(f"Average LPIPS: {average_lpips}")
    else:
        print("No valid image pairs found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average SSIM, PSNR, and LPIPS between two folders of images.")
    parser.add_argument('--test_folder', type=str, required=True, help="Path to the folder containing test images.")
    parser.add_argument('--original_folder', type=str, required=True,
                        help="Path to the folder containing original images.")
    parser.add_argument('--scale', type=float, default=0.375, help="Scaling factor for the original images.")

    args = parser.parse_args()
    main(args)


