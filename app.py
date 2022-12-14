import tkinter as tk
from tkinter import filedialog, Text
from tkinter.messagebox import showinfo
import os

import librosa as lr
import numpy as np
from pydub import AudioSegment
#import ffmpeg
import json

lst = []
data = []

# def prep_audio_files(s_name):
#     song = AudioSegment.from_file(s_name)
#     if len(song) <= 15000:
#         print("test")
#     song = song[30000:45000]
#     song.export(s_name)
#     return song

def main():
    application2()
    return 0 

def use_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("WAV Files", "*.wav"), ("all files", "*.*")))
    data.append(filename)
    print(filename)
    more = "audio file:  "
    for i in data:
        for j in lst:
            txt = tk.Label(j,text=more + i, bg="gray")
            txt.pack()

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