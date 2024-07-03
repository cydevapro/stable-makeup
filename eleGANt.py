import os
import numpy as np
from PIL import Image
import torch

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args


def transfer(id_image_path, makeup_image_path, output_path):
    class Args:
        def __init__(self):
            self.name = 'demo'
            self.save_path = 'result'
            self.load_path = 'ckpts/sow_pyramid_a5_e3d2_remapped.pth'
            self.source_dir = "assets/images/non-makeup"
            self.reference_dir = "assets/images/makeup"
            self.gpu = 'cpu'

    args = Args()
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    config = get_config()

    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path)

    imgA = Image.open(id_image_path).convert('RGB')
    imgB = Image.open(makeup_image_path).convert('RGB')

    imgA = imgA.resize((512, 512))
    imgB = imgB.resize((512, 512))

    result = inference.transfer(imgA, imgB, postprocess=True)

    imgA = np.array(imgA)
    imgB = np.array(imgB)
    h, w, _ = imgA.shape
    result = result.resize((h, w))
    result = np.array(result)
    vis_image = np.hstack((imgA, imgB, result))
    Image.fromarray(vis_image.astype(np.uint8)).save(output_path)
