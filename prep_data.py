import librosa as lr
import pandas as pd
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
    SAMPLE_RATE = 22050
    NUM_SLICES = 10
    TOTAL_SAMPLES = 15 * SAMPLE_RATE
    SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / NUM_SLICES)
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
            for s in range(NUM_SLICES):
                start = SAMPLES_PER_SLICE * s
                finish = start + SAMPLES_PER_SLICE
                mfcc = lr.feature.mfcc(y=y[start:finish], sr=sr , n_mfcc=13)
                mfcc = mfcc.T
                mfcc_dict['mfcc'].append(mfcc.tolist())
                mfcc_dict['labels'].append(dir)

    pd.DataFrame(mfcc_dict).to_csv('data.csv')
    #with open('data.json', 'w') as fp:
        #json.dump(mfcc_dict, fp, indent=4)  
        #fp.close()

if __name__ == "__main__":
    main()