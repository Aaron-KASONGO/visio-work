import shutil
import os
import requests
import torch
from gtts import gTTS

from typing import Union
from io import BytesIO

from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from transformers import pipeline

import scipy


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/detect-object/")
async def detect_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    reponse_list = image_to_text(image)
    reponse_text = ""
    if (reponse_list):
        reponse_text = reponse_list[0]["generated_text"]
    print(type(reponse_text))
    tts = gTTS(text=reponse_text, slow=False)
    tts.save("audio.mp3")
    
    # scipy.io.wavfile.write("bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])
    return FileResponse("audio.mp3", media_type='audio/mp3', headers={"Content-Disposition": "attachment; filename=audio.mp3"})
