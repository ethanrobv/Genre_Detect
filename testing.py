import librosa as lr
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment
import ffmpeg

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
    for dir in os.listdir('./audio'):
        for file in os.listdir('./audio/' + dir):
            y, sr = lr.load('./audio/' + dir + '/' + file)
            #lr.display.waveshow(y, sr=sr)

            D = lr.stft(y)
            S_db = lr.amplitude_to_db(np.abs(D), ref=np.max)
            lr.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar()

            plt.show()

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