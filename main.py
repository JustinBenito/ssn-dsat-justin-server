from fastapi import FastAPI, File, UploadFile
import parselmouth
import os

import pandas as pd
import numpy as np
import librosa
import librosa.display
import opensmile
from Signal_Analysis.features.signal import *
import matplotlib
from PIL import Image


from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:5173",
    "*"
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
basedir = os.path.abspath(os.path.dirname(__file__))

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/extract")
async def extract_file(file: UploadFile):
    file_name = os.getcwd()+"/audios/"+file.filename.replace(" ", "-")
    try:
        os.mkdir("audios")
        print(os.getcwd())

    except Exception as e:
        print(e) 
        print("Error",file_name)

    with open(file_name,'wb+') as f:
        f.write(file.file.read())
        f.close()
        
        y, sr = librosa.load(file_name)
      
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        op = smile.process_file(file_name)
        sound = parselmouth.Sound(file_name)

        energy = sound.to_intensity()

        sound_vals=list(np.array(sound.values)[0])

        energy_plot=list(np.array(energy.values)[0])
        s = op.to_dict('records')
        opsmile = s[0]

        f0 =  get_F_0( y, sr )
        pitch = sound.to_pitch()
        f0 = f0[0]    
        feat={
                'energy_plot': energy_plot,
                
                'sound_plot': sound_vals,

                'pitch_plot': list(np.array(pitch.selected_array['frequency'])),
                
                'filename': file_name
        } 
    return {"features":feat}
