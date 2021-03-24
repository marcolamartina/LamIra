#!/bin/bash
# pip
sudo apt install python3-pip

# matplotlib
pip3 install matplotlib

# git
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install git
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install git
fi

# Speech_to_text
pip3 install SpeechRecognition
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install portaudio --HEAD
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install portaudio19-dev
    sudo apt-get install python3-pyaudio -y
    sudo apt-get install libasound2-dev
    sudo apt-get install ffmpeg
fi
pip3 install pyaudio


# Text_to_speech
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install ffmpeg
fi
pip3 install gtts
pip3 install pydub
# sudo nano /usr/share/alsa/alsa.conf
# if there are errors, change the association with default for the row with error

# Intent_classification
if [[ "$OSTYPE" == "darwin"* ]]; then
    pip install torch torchvision torchaudio
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh;
    source $HOME/.cargo/env
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    pip3 install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
fi
pip3 install transformers==3

# Kinect
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install libfreenect
    freenect-glview # for test
    # install CMake from https://cmake.org/download/
    
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install git cmake build-essential libusb-1.0-0-dev
    sudo apt-get install libfreenect-bin
fi

pip3 install cython

git clone https://github.com/OpenKinect/libfreenect
cd libfreenect
mkdir build
cd build
cmake -L .. # -L lists all the project options
make
cmake .. -DBUILD_PYTHON3=ON
make
cmake .. -DBUILD_REDIST_PACKAGE=OFF
cd ../wrappers/python
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LD_LIBRARY_PATH=/usr/local/lib
fi
sudo python3 setup.py install

# replace tp_print with tp_vectorcall_offset in libfreenect/wrappers/python/freenect.c






