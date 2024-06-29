import os

import ngrok
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from diffusers import DDIMScheduler, ControlNetModel
from diffusers.utils import load_image
from detail_encoder.encoder_plus import detail_encoder
from pipeline_sd15 import StableDiffusionControlNetPipeline
from spiga_draw import *
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from facelib import FaceDetector
from PIL import Image

app = FastAPI()


def run_ngrok(port):
    listener = ngrok.forward(port,'tcp' ,authtoken="2YtApGcOINFy3F3oA7T0uxIkIIn_nie3EqfnyfaoyMieAiZC")
    # Output ngrok url to console
    return listener.url()


ngrok_url = run_ngrok(5001).replace('tcp','http')
print(f" * Running on {ngrok_url}")

# Configure upload and output folders
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mount static files for serving processed images
app.mount("/static", StaticFiles(directory=OUTPUT_FOLDER), name="static")

# Initialize models
processor = SPIGAFramework(ModelConfig("300wpublic"))
detector = FaceDetector(weight_path="./models/mobilenet0.25_Final.pth")

model_id = "runwayml/stable-diffusion-v1-5"
makeup_encoder_path = "./models/stablemakeup/pytorch_model.bin"
id_encoder_path = "./models/stablemakeup/pytorch_model_1.bin"
pose_encoder_path = "./models/stablemakeup/pytorch_model_2.bin"

Unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
id_encoder = ControlNetModel.from_unet(Unet)
pose_encoder = ControlNetModel.from_unet(Unet)
makeup_encoder = detail_encoder(Unet, "openai/clip-vit-large-patch14", "cuda", dtype=torch.float32)

makeup_state_dict = torch.load(makeup_encoder_path, map_location=torch.device('cuda'))
id_state_dict = torch.load(id_encoder_path, map_location=torch.device('cuda'))
pose_state_dict = torch.load(pose_encoder_path, map_location=torch.device('cuda'))

id_encoder.load_state_dict(id_state_dict, strict=False)
pose_encoder.load_state_dict(pose_state_dict, strict=False)
makeup_encoder.load_state_dict(makeup_state_dict, strict=False)

id_encoder.to("cuda")
pose_encoder.to("cuda")
makeup_encoder.to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    unet=Unet,
    controlnet=[id_encoder, pose_encoder],
    torch_dtype=torch.float32).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


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


def transfer(id_image_path, makeup_image_path, output_path):
    id_image = load_image(id_image_path).resize((512, 512))
    makeup_image = load_image(makeup_image_path).resize((512, 512))
    pose_image = get_draw(id_image, size=512)

    guidance_scale = 1.1  # Adjust scale
    num_inference_steps = 50  # Number of inference steps

    result_img = makeup_encoder.generate(id_image=[id_image, pose_image],
                                         makeup_image=makeup_image,
                                         pipe=pipe,
                                         guidance_scale=guidance_scale,
                                         num_inference_steps=num_inference_steps)

    result_img.save(output_path)


@app.post("/transfer/v1/")
async def transfer_endpoint(id_image: UploadFile = File(...), makeup_image: UploadFile = File(...)):
    if not (is_image_file(id_image.filename) and is_image_file(makeup_image.filename)):
        raise HTTPException(status_code=400, detail="Invalid image files")

    id_image_path = os.path.join(UPLOAD_FOLDER, id_image.filename)
    makeup_image_path = os.path.join(UPLOAD_FOLDER, makeup_image.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{id_image.filename}_{makeup_image.filename}")

    with open(id_image_path, "wb") as f:
        f.write(await id_image.read())

    with open(makeup_image_path, "wb") as f:
        f.write(await makeup_image.read())

    transfer(id_image_path, makeup_image_path, output_path)

    processed_url = f"/static/{os.path.basename(output_path)}"
    return JSONResponse(content={"result_image": processed_url}, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="debug")
