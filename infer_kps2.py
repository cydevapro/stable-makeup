import os
import torch
from PIL import Image
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from diffusers import DDIMScheduler, ControlNetModel
from diffusers.utils import load_image
from detail_encoder.encoder_plus import detail_encoder
from pipeline_sd15 import StableDiffusionControlNetPipeline
from spiga_draw import *
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from facelib import FaceDetector

processor = SPIGAFramework(ModelConfig("300wpublic"))
detector = FaceDetector(weight_path="./models/mobilenet0.25_Final.pth")


def get_draw(pil_img, size):
    spigas = spiga_process(pil_img, detector)
    if not spigas:
        width, height = pil_img.size
        black_image_pil = Image.new('RGB', (width, height), color=(0, 0, 0))
        return black_image_pil
    else:
        spigas_faces = spiga_segmentation(spigas, size=size)
        return spigas_faces


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def concatenate_images(image_files, output_file):
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)


model_id = "runwayml/stable-diffusion-v1-5"  # your sdv1-5 path
makeup_encoder_path = "./models/stablemakeup/pytorch_model.bin"
id_encoder_path = "./models/stablemakeup/pytorch_model_1.bin"
pose_encoder_path = "./models/stablemakeup/pytorch_model_2.bin"
Unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cpu")

id_encoder = ControlNetModel.from_unet(Unet)
pose_encoder = ControlNetModel.from_unet(Unet)
makeup_encoder = detail_encoder(Unet, "openai/clip-vit-large-patch14", "cpu", dtype=torch.float32)
makeup_state_dict = torch.load(makeup_encoder_path, map_location=torch.device('cpu'))
id_state_dict = torch.load(id_encoder_path, map_location=torch.device('cpu'))
id_encoder.load_state_dict(id_state_dict, strict=False)
pose_state_dict = torch.load(pose_encoder_path, map_location=torch.device('cpu'))
pose_encoder.load_state_dict(pose_state_dict, strict=False)
makeup_encoder.load_state_dict(makeup_state_dict, strict=False)
id_encoder.to("cpu")
pose_encoder.to("cpu")
makeup_encoder.to("cpu")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    unet=Unet,
    controlnet=[id_encoder, pose_encoder],
    torch_dtype=torch.float32).to("cpu")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


def infer():
    id_folder = "./test_imgs/id"
    makeup_folder = "./test_imgs/makeup"
    out_folder = "./output/new1"
    os.makedirs(out_folder, exist_ok=True)
    for name in os.listdir(id_folder):
        if not is_image_file(name):
            continue
        id_image = load_image(os.path.join(id_folder, name)).resize((512, 512))
        for mu in os.listdir(makeup_folder):
            if not is_image_file(mu):
                continue
            makeup_image = load_image(os.path.join(makeup_folder, mu)).resize((512, 512))
            pose_image = get_draw(id_image, size=512)

            guidance_scale = 3.1#1.01
            num_inference_steps = 50
            result_img = makeup_encoder.generate(id_image=[id_image, pose_image],
                                                 makeup_image=makeup_image,
                                                 pipe=pipe,
                                                 guidance_scale=guidance_scale,
                                                 num_inference_steps=num_inference_steps)
            result_img.save(os.path.join(out_folder, f"{name.split('.')[0]}_{mu.split('.')[0]}.png"))


if __name__ == '__main__':
    infer()
