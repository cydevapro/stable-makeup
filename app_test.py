import subprocess
import time
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter

import ngrok

from eleGANt import transfer


app = FastAPI()

UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'

PROCESSED_FOLDER = 'static/processed_images'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def run_ngrok(port):
    listener = ngrok.forward(port,'tcp' ,authtoken="2YtApGcOINFy3F3oA7T0uxIkIIn_nie3EqfnyfaoyMieAiZC")
    # Output ngrok url to console
    return listener.url()


ngrok_url = run_ngrok(5001).replace('tcp','http')
print(f" * Running on {ngrok_url}")

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp'])


@app.post("/transfer/v1/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    # Process the image
    image = Image.open(filepath)
    processed_image = image.filter(ImageFilter.BLUR)
    processed_filename = f"processed_{file.filename}"
    processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
    processed_image.save(processed_filepath)

    # Return the URL of the processed image
    processed_url = f"{ngrok_url}/static/processed_images/{processed_filename}"
    return JSONResponse(content={"processed_image_url": processed_url}, status_code=200)


@app.post("/transfer/v2/")
async def transfer_endpoint(id_image: UploadFile = File(...), makeup_image: UploadFile = File(...)):
    if not (is_image_file(id_image.filename) and is_image_file(makeup_image.filename)):
        raise HTTPException(status_code=400, detail="Invalid image files")

    id_image_path = os.path.join(UPLOAD_FOLDER, id_image.filename)
    makeup_image_path = os.path.join(UPLOAD_FOLDER, makeup_image.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{id_image.filename}_{makeup_image.filename}.png")

    with open(id_image_path, "wb") as f:
        f.write(await id_image.read())

    with open(makeup_image_path, "wb") as f:
        f.write(await makeup_image.read())

    transfer(id_image_path, makeup_image_path, output_path)

    processed_url = f"{ngrok_url}/static/{os.path.basename(output_path)}"
    return JSONResponse(content={"result_img": processed_url}, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="debug")
