import torch
from PIL import Image
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from pipeline_sd15 import StableDiffusionControlNetPipeline
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, ControlNetModel
from detail_encoder.encoder_plus import detail_encoder
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


# Initialize the model
model_id = "runwayml/stable-diffusion-v1-5"  # your sd1.5 model path
base_path = "./models/stablemakeup"
makeup_encoder_path = base_path + "/pytorch_model.bin"
id_encoder_path = base_path + "/pytorch_model_1.bin"
pose_encoder_path = base_path + "/pytorch_model_2.bin"

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

pose_encoder.to("cpu")
id_encoder.to("cpu")
makeup_encoder.to("cpu")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    unet=Unet,
    controlnet=[id_encoder, pose_encoder],
    torch_dtype=torch.float32).to("cpu")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# Define your model call function
def model_call(id_image_path, makeup_image_path, num):
    img_size =512
    id_image = Image.open(id_image_path).convert('RGB')
    makeup_image = Image.open(makeup_image_path).convert('RGB')
    id_image = id_image.resize((img_size, img_size))
    makeup_image = makeup_image.resize((img_size, img_size))
    pose_image = get_draw(id_image, size=img_size)
    result_img = makeup_encoder.generate(id_image=[id_image, pose_image], makeup_image=makeup_image, guidance_scale=num,
                                         pipe=pipe)
    return result_img


# Example usage:
id_image_path = "test_imgs/id/1.jpg"
makeup_image_path = "test_imgs/makeup/1.png"
guidance_scale = 1.05  # example value

result_image = model_call(id_image_path, makeup_image_path, guidance_scale)
result_image.show()  # Display or save the resulting image as needed
