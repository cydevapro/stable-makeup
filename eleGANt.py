import os
import numpy as np
from PIL import Image
import torch

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args
from upscale import upscale_image


def transfer_v2(id_image_path, makeup_image_path, output_path):
    class Args:
        def __init__(self):
            self.save_path = 'result'
            self.load_path = 'ckpts/sow_pyramid_a5_e3d2_remapped.pth'
            self.gpu = 'cuda'

    args = Args()
    args.device = torch.device(args.gpu)

    config = get_config()
    inference = Inference(config, args, args.load_path)

    imgA = Image.open(id_image_path).convert('RGB')
    imgB = Image.open(makeup_image_path).convert('RGB')

    # imgA = Image.open('test_imgs/id/a.jpg').convert('RGB')
    # imgB = Image.open('test_imgs/makeup/1.png').convert('RGB')

    imgA = imgA.resize((361, 361))
    imgB = imgB.resize((361, 361))

    result = inference.transfer(imgA, imgB, postprocess=True)

    imgA = np.array(imgA)
    imgB = np.array(imgB)
    h, w, _ = imgA.shape
    result = result.resize((h, w))
    result = np.array(result)
    vis_image = np.hstack((imgA, imgB, result))
    Image.fromarray(result.astype(np.uint8)).save(output_path)

    upscale_image(output_path,output_path)
