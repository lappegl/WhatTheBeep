# What The Beep
## video audio censoring with openai-whisper transcription

Are you a content creator or pod-caster uploading to platforms that are very sensitive when certain topics are discussed? *cough*-*cough*-*YouTube*-*cough*

This project aims to automate audio censoring / bleeping of keywords, saving you much time scanning through your content to manually identify and censor your content.



## How it works

The main idea is this:
* use openai-whisper to transcribe audio, with timestamps for each spoken word
* identify any targeted "no-no" words in the transcription after defining the list in **flag_words.txt**
* use the timestamps of the identified "no-no" words to overlay a beep or silence at the correct time

Provided a .mp4 input file, audio will be extracted, transcribed, censored, and the censored audio will be overlaid onto a new "_overlaid.mp4"

Provided a .wav file, audio will be transcribed and a "_censored.wav" file will be created.

A "_transcription.csv" file is also created with timestamps of the openai-whisper results for manual review.


## Demo

[![See Demo Video](https://img.youtube.com/vi/W7O0dOXlO8A/0.jpg)](https://www.youtube.com/watch?v=W7O0dOXlO8A)

## Requirements / Installation Guide

### ffmpeg -- https://ffmpeg.org/download.html

Remember to check that your environment variables include the directory of your installed **ffmpeg.exe**

### CUDA Toolkit (recommended, for GPU acceleration of the whisper audio transcription)

I did my development and testing on Windows 10 with CUDA Toolkit 11.8.

12.1 should also work, but this will affect how you install pytorch.

CUDA Toolkit can be downloaded from here: https://developer.nvidia.com/cuda-11-8-0-download-archive

### Notes on PyTorch

Be sure to install PyTorch BEFORE openai-whisper. As torch is a dependency of whisper, it may automatically be downloaded, but installing PyTorch manually assures you get the right version for your CUDA version.

In requirements.txt I specified index-url https://download.pytorch.org/whl/cu118 for CUDA 11.8

If you installed CUDA 12.1, you'd use https://download.pytorch.org/whl/cu121

See PyTorch install page to see instructions for your specific setup: https://pytorch.org/get-started/locally/

### Verifying CUDA and PyTorch installation

Create and run a python script:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```
This should show **True** if CUDA and an appropriate PyTorch version were correctly installed.

GPU acceleration is not required for the openai-whisper transcription, but it is much faster.


