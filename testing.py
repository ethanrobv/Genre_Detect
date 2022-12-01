import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment
import ffmpeg
import json

# truncate all songs to 15 seconds
def prep_audio_files():
    for dir in os.listdir('./audio'):
        if dir.startswith('.'):
            continue
        for file in os.listdir('./audio/' + dir):
            if (file.startswith('.')):
                continue
            print(file)
            song = AudioSegment.from_file('./audio/' + dir + '/' + file)
            if len(song) <= 15000:
                continue
            song = song[30000:45000]
            song.export('./audio/' + dir + '/' + file)

def main():
    prep_audio_files()
    mfcc_dict = {}
    mfcc_dict['labels'] = []
    mfcc_dict['mfcc'] = []
    for dir in os.listdir('./audio'):
        if dir.startswith('.'):
            continue
        for file in os.listdir('./audio/' + dir):
            if (file.startswith('.')):
                continue
            y, sr = lr.load('./audio/' + dir + '/' + file)
            #lr.display.waveshow(y, sr=sr)
            """
            D = lr.stft(y)
            S_db = lr.amplitude_to_db(np.abs(D), ref=np.max)
            lr.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar()

            plt.show()
            """
            mfcc = lr.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc = mfcc.T
            mfcc_dict['labels'].append(dir)
            mfcc_dict['mfcc'].append(mfcc.tolist())

    with open('data.json', 'w') as fp:
        json.dump(mfcc_dict, fp, indent=4)  
        fp.close()

    """
    song = AudioSegment.from_file("./audio/onclassical_demo_demicheli_geminiani_pieces_allegro-in-f-major_small-version.wav")
    song = song[:15000]
    song.export("./audio/onclassical_demo_demicheli_geminiani_pieces_allegro-in-f-major_small-version.wav")

    audio, sr = lr.load('./audio/onclassical_demo_demicheli_geminiani_pieces_allegro-in-f-major_small-version.wav')

    # plot the audio
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(audio, sr=sr)

    # show spectrogram
    D = lr.stft(audio)
    S_db = lr.amplitude_to_db(abs(D), ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

    plt.show()
    """

if __name__ == "__main__":
    main()