import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image

sys.path.append('.')

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args


def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path)

    imgA = Image.open('test_imgs/id/a.jpg').convert('RGB')
    imgB = Image.open('test_imgs/makeup/1.png').convert('RGB')

    imgA = imgA.resize((512, 512))
    imgB = imgB.resize((512, 512))

    result = inference.transfer(imgA, imgB, postprocess=True)

    imgA = np.array(imgA)
    imgB = np.array(imgB)
    h, w, _ = imgA.shape
    result = result.resize((h, w));
    result = np.array(result)
    vis_image = np.hstack((imgA, imgB, result))
    save_path = os.path.join(args.save_folder, f"result.png")
    Image.fromarray(vis_image.astype(np.uint8)).save(save_path)


if __name__ == "__main__":
    tnp = 'ckpts/sow_pyramid_a5_e3d2_remapped.pth'

    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model",
                        default=tnp)

    parser.add_argument("--source-dir", type=str, default="assets/images/non-makeup")
    parser.add_argument("--reference-dir", type=str, default="assets/images/makeup")
    parser.add_argument('--gpu', type=str, default='cpu', help='GPU device id (e.g., 0 or 1). Use "cpu" to run on CPU.')

    args = parser.parse_args()
    args.gpu = args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    config = get_config()
    main(config, args)
