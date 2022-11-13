import librosa as lr
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment
import ffmpeg

# truncate all songs to 15 seconds
def prep_audio_files():
    song = AudioSegment.from_wav("./audio/onclassical_demo_demicheli_geminiani_pieces_allegro-in-f-major_small-version.wav")
    song.export('./audio/onclassical_demo_demicheli_geminiani_pieces_allegro-in-f-major_small-version.wav')

def main():
    
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

if __name__ == "__main__":
    main()