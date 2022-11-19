#!bin/bash

unzip ./audio_backup.zip
sudo rm -r ./__MACOSX
sudo rm -r ./audio
sudo mv ./backup ./audio