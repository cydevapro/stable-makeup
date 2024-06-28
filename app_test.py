import subprocess
import time
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter

import ngrok

app = FastAPI()

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def run_ngrok(port):
    listener = ngrok.forward(port,'tcp' ,authtoken="2YtApGcOINFy3F3oA7T0uxIkIIn_nie3EqfnyfaoyMieAiZC")
    # Output ngrok url to console
    return listener.url()


ngrok_url = run_ngrok(5001).replace('tcp','http')
print(f" * Running on {ngrok_url}")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="debug")
