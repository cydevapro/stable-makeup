import torch
from flask import Flask, request, jsonify, url_for

from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from diffusers import DDIMScheduler, ControlNetModel
from diffusers.utils import load_image
from detail_encoder.encoder_plus import detail_encoder
from pipeline_sd15 import StableDiffusionControlNetPipeline
from spiga_draw import *
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from facelib import FaceDetector

app = Flask(__name__)

# Khởi tạo các biến và mô hình
processor = SPIGAFramework(ModelConfig("300wpublic"))
detector = FaceDetector(weight_path="./models/mobilenet0.25_Final.pth")

model_id = "runwayml/stable-diffusion-v1-5"  # Đường dẫn đến model Stable Diffusion v1.5
makeup_encoder_path = "./models/stablemakeup/pytorch_model.bin"
id_encoder_path = "./models/stablemakeup/pytorch_model_1.bin"
pose_encoder_path = "./models/stablemakeup/pytorch_model_2.bin"
Unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cpu")

id_encoder = ControlNetModel.from_unet(Unet)
pose_encoder = ControlNetModel.from_unet(Unet)
makeup_encoder = detail_encoder(Unet, "openai/clip-vit-large-patch14", "cpu", dtype=torch.float32)
makeup_state_dict = torch.load(makeup_encoder_path, map_location=torch.device('cpu'))
id_state_dict = torch.load(id_encoder_path, map_location=torch.device('cpu'))
pose_state_dict = torch.load(pose_encoder_path, map_location=torch.device('cpu'))
id_encoder.load_state_dict(id_state_dict, strict=False)
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

    guidance_scale = 3.1  # Điều chỉnh scale
    num_inference_steps = 50  # Số lần lặp inference

    result_img = makeup_encoder.generate(id_image=[id_image, pose_image],
                                         makeup_image=makeup_image,
                                         pipe=pipe,
                                         guidance_scale=guidance_scale,
                                         num_inference_steps=num_inference_steps)

    result_img.save(output_path)


@app.route('/transfer/v1/', methods=['POST'])
def transfer_endpoint():
    if 'id_image' not in request.files or 'makeup_image' not in request.files:
        return jsonify({"error": "Missing image files"}), 400

    id_image_file = request.files['id_image']
    makeup_image_file = request.files['makeup_image']

    id_image_path = os.path.join(app.config['UPLOAD_FOLDER'], id_image_file.filename)
    makeup_image_path = os.path.join(app.config['UPLOAD_FOLDER'], makeup_image_file.filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'],
                               f"output_{id_image_file.filename}_{makeup_image_file.filename}")

    id_image_file.save(id_image_path)
    makeup_image_file.save(makeup_image_path)

    transfer(id_image_path, makeup_image_path, output_path)

    # Trả về URL của ảnh đã xử lý
    processed_url = url_for('static', filename=f'processed_images/{os.path.basename(output_path)}', _external=True)
    return jsonify({"result_image": processed_url}), 200


if __name__ == '__main__':
    app.run(debug=True)
