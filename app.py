import tkinter as tk
from tkinter import filedialog, Text
from tkinter.messagebox import showinfo
import os

import tensorflow as tf

import librosa as lr
import numpy as np
from pydub import AudioSegment
import ffmpeg
import json
import model as md
from model import CustomDenseVariational
import pandas as pd

lst = []
data_list = []

# def prep_audio_files(s_name):
#     song = AudioSegment.from_file(s_name)
#     if len(song) <= 15000:
#         print("test")
#     song = song[30000:45000]
#     song.export(s_name)
#     return song

SAMPLE_RATE = 22050
NUM_SLICES = 10
TOTAL_SAMPLES = 15 * SAMPLE_RATE
SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / NUM_SLICES)
file = "./SnapSave.io - Southern Cross (128 kbps).mp3"

def main():
    application2()
    return 0 

def use_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("MP3 Files", "*.mp3"), ("all files", "*.*")))
    data_list.append(filename)
    file = filename
    print(filename)
    more = "audio file:  "
    for i in data_list:
        for j in lst:
            txt = tk.Label(j,text=more + i, bg="gray")
            txt.pack()
    
def make_prediction():
    with tf.keras.utils.CustomObjectScope({'CustomDenseVariational': CustomDenseVariational}):
        model = md.create_model()
        model.load_weights('model_weights.h5')
        y, sr = lr.load(data_list[0])
        mfcc_dict = {}
        mfcc_dict['mfcc'] = []
        mfcc = None
        for s in range(NUM_SLICES):
                start = SAMPLES_PER_SLICE * s
                finish = start + SAMPLES_PER_SLICE
                mfcc = lr.feature.mfcc(y=y[start:finish], sr=sr , n_mfcc=13)
                mfcc = mfcc.T
                mfcc_dict['mfcc'].append(mfcc.tolist())
        mfcc_df = pd.DataFrame(mfcc_dict)
        mfcc_df.to_csv('datum.csv', index=False)
        # remove all quotation marks from data.csv
        with open('datum.csv', 'r') as f:
            lines = f.readlines()
        with open('datum.csv', 'w') as f:
            for line in lines:
                f.write(line.replace('"', ''))
        
        # remove all brackets and double brackets from data.csv
        with open('datum.csv', 'r') as f:
            lines = f.readlines()
        with open('datum.csv', 'w') as f:
            for line in lines:
                f.write(line.replace('[', ''))
        
        with open('datum.csv', 'r') as f:
            lines = f.readlines()
        with open('datum.csv', 'w') as f:
            for line in lines:
                f.write(line.replace(']', ''))
        
        with open('datum.csv', 'r') as f:
            lines = f.readlines()
        with open('datum.csv', 'w') as f:
            for line in lines:
                f.write(line.replace('[[', ''))
        
        with open('datum.csv', 'r') as f:
            lines = f.readlines()
        with open('datum.csv', 'w') as f:
            for line in lines:
                f.write(line.replace(']]', ''))
        
        data=np.loadtxt(fname='./data.csv',delimiter=',',dtype=float,skiprows=1)
        # delete first two columns
        data = np.delete(data, [0, 1], axis=1)
    

        data = np.asarray(data)


        prediction = model.predict(data)
        predicted_classes = np.argmax(prediction, axis=-1)
        output = ""
        tally_zero = 0
        tally_one = 0
        for pred in predicted_classes:
            if pred == 0:
                tally_zero += 1
            else:
                tally_one += 1
        if tally_zero > tally_one:
            output = "Hip-Hop"
        else:
            output = "Rock"

    
        tk.Message(lst[0], text="Prediction: " + str(output)).pack()

def application2():
    root = tk.Tk()
    canvas = tk.Canvas(root, height=700, width=700, bg="#263D42")
    canvas.create_text(250, 25, text="CptS 437 Project", fill="black", font=('Helvetica 15 bold'))
    canvas.pack()
    frame = tk.Frame(root, bg="white")
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    lst.append(frame)
    open_f = tk.Button(root, text="Open File", padx=5, pady=5, fg="white", bg="#263D42", command=use_file)
    open_f.pack()

    predict_button = tk.Button(root, text="Predict", padx=5, pady=5, fg="white", bg="#263D42", command=make_prediction)
    predict_button.pack()


    root.mainloop()

    return None


if __name__ == "__main__":
    main()

""" def application():
    root = tk.Tk()

    canvas = tk.Canvas(root, height=400, width=400, bg="#263D42")
    canvas.create_text(200, 25, text="CptS 437 Project", fill="black", font=('Helvetica 15 bold'))
    canvas.pack()

    frame = tk.Frame(root, bg="white")
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    
    for widget in frame.winfo_children():
        widget.destroy()
    open_file(root)
    g = "audio file: "
    for i in range(0, len(data)):
        g + data[i]
    txt = tk.Label(frame, text=g,bg="gray")
    txt.pack()
    root.mainloop()

    return None 

def proccess_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("WAV Files", "*.wav"), ("all files", "*.*")))
    print(filename)
    data.append(filename)
    e = showinfo(title="Selected Files: ", message=filename)

def open_file(fname):
    open_f = tk.Button(fname, text="Open File", padx=5, pady=5, fg="white", bg="#263D42", command=proccess_file)
    open_f.pack() """