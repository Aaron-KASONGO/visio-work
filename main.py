import shutil
import os
import requests
import torch

from typing import Union
from io import BytesIO

from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from transformers import pipeline



app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/detect-object/")
async def detect_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    reponse = image_to_text(image)
    return {"info": f"file '{reponse}'"}