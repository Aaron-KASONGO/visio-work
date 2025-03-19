import shutil
import os
import requests
import torch
import soundfile as sf
from gtts import gTTS

from typing import Union
from io import BytesIO

from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from transformers import pipeline

import scipy


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/detect-object/")
async def detect_file(file: UploadFile = File(...)):
    contents = await file.read()
    reponse_list = image_to_text(image)
    reponse_text = ""
    if (reponse_list):
        reponse_text = reponse_list[0]["generated_text"]
    print(type(reponse_text))
    tts = gTTS(text=reponse_text, slow=False)
    tts.save("audio.mp3")
    
    # scipy.io.wavfile.write("bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])
    return {"info": f"file '{reponse_text}'"}
