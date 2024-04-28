from ffmpeg import FFmpeg
import whisper
from pathlib import Path
from datetime import datetime
import pandas as pd
import csv
import os
from string import ascii_letters, digits
import numpy as np
import pydub
from moviepy.editor import VideoFileClip, AudioFileClip


CODEC = 'pcm_s16le'  # "pcm_s16le" for better quality, or "ac3"
BEEP = True  # set to False to instead override censored words with silence
BEEP_VOLUME = 0.1  # range from [0, 1] but quite sensitive.
BEEP_SIN_FREQUENCY = 1000  # lower = lower tone, higher = higher tone. 800-1000 seems fine.


def load_flagwords():
    words = []
    with open('flag_words.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'):
            continue
        line = line.strip().lower()
        if line != '':
            words.append(line)
    return words


CENSOR_WORDS = load_flagwords()

WHISPER_MODEL = 'medium.en'

ALLOWED_CHARS = ascii_letters + digits + "'"  # keep only these characters from transcribed words for comparison

FRAMERATE = None  # detected and set automatically


def extract_audio(mp4_file_path):
    mp4_file_path = str(Path(mp4_file_path).absolute())
    print('Extracting audio from file:', mp4_file_path)
    audio_file_path = mp4_file_path.replace('.mp4', '.wav')
    if not os.path.exists(audio_file_path):
        FFmpeg().input(mp4_file_path).output(audio_file_path, acodec=CODEC).execute()
    else:
        print('Extracted audio already exists:', audio_file_path)
    return audio_file_path


def transcribe(audio_file, model='medium.en'):
    model = whisper.load_model(model)
    print('Transcribing audio...')
    start = datetime.now()
    result = model.transcribe(audio_file, word_timestamps=True)
    end = datetime.now()
    print('Transcribing finished in', end - start)
    return result


def convert_to_timestamp(seconds):
    hours = seconds // 3600
    hours = f"00{int(hours)}"[-2:]
    minutes = (seconds % 3600) // 60
    minutes = f"00{int(minutes)}"[-2:]
    seconds = seconds % 60
    seconds = f"00{int(seconds)}"[-2:]
    return f"{hours}:{minutes}:{seconds}"


def clean_text(text):
    result = [c for c in text if c in ALLOWED_CHARS]
    return ''.join(result).lower()


def create_transcription_csv(transcription, save_path):
    word_data = []
    for segment in transcription['segments']:
        for word_dict in segment['words']:
            word_data.append(word_dict)

    df = pd.DataFrame(word_data)
    del df['probability']
    df['time'] = df['start'].apply(convert_to_timestamp)
    # reorder columns
    df_final = pd.DataFrame()
    for col in ['word', 'start', 'end', 'time']:
        df_final[col] = df[col]

    df_final['text'] = df_final['word'].apply(clean_text)
    df_final['start'] = df_final['start'].apply(lambda x: round(x, 2))
    df_final['end'] = df_final['end'].apply(lambda x: round(x, 2))
    df_final['flagged'] = df_final['text'].isin(CENSOR_WORDS).astype(int)
    df_final.to_csv(save_path, quoting=csv.QUOTE_ALL, index=False)
    return df_final


def make_beep(duration, sampling_rate, sine_freq=BEEP_SIN_FREQUENCY):
    # generate samples, note conversion to float32 array
    samples = (np.sin(2 * np.pi * np.arange(sampling_rate * duration) * sine_freq / sampling_rate)).astype(np.float32)
    max_val = np.iinfo(np.int16).max
    scaled_samples = (max_val * BEEP_VOLUME * samples).astype(np.int16)
    audio_segment = pydub.AudioSegment(
        data=scaled_samples.tobytes(),
        sample_width=2,  # 16-bit audio.
        frame_rate=sampling_rate,
        channels=1  # Mono. Do not set to 2 without actually creating new samples, or audio will play in just 1 ear
    )
    return audio_segment


def create_transcription(path_to_audio_file):
    transcription_file = path_to_audio_file.replace('.wav', '_transcription.csv')
    transcription = transcribe(path_to_audio_file)
    df = create_transcription_csv(transcription, path_to_audio_file.replace('.wav', '_transcription.csv'))
    return df, transcription_file


def censor_audio(path_to_audio_file, beep=True, transcription_file=None):
    """
    Parameters
    ----------
    path_to_audio_file
    beep: True to overlay with a beep, False to overlay with silence
    transcription_file: in case audio is already transcribed and want to make edits to the transcription file, such as:
        - corrections to the transcription (column: text)
        - corrections to timings (columns: start + end)
        - manual flagging of additional words (column: flag)

    Returns
    -------
    path to censored audio file
    """
    if transcription_file is None:
        df, transcription_file = create_transcription(path_to_audio_file)
    else:
        df = pd.read_csv(transcription_file, quoting=csv.QUOTE_ALL)

    audio = pydub.AudioSegment.from_wav(path_to_audio_file)
    mx = len(audio) - 1
    global FRAMERATE
    FRAMERATE = audio.frame_rate

    # identify start + end of each segment to override
    silence_segments = []
    for i, row in df.iterrows():
        if row['flagged'] == 1:
            # +/- buffer to cover any potential timing error
            start_ms = row['start'] * 1000 - 40
            end_ms = min([row['end'] * 1000 + 70, mx])  # prevents new segment from extending length of original audio
            silence_segments.append((start_ms, end_ms))

    modified_segments = []
    prior = 0
    if silence_segments:
        # censor each segment and reassemble complete audio
        for i, ss in enumerate(silence_segments):
            start, end = ss
            before = audio[prior:start]
            duration = end - start
            modified_segments.append(before)
            if beep:
                censor = make_beep(duration=duration / 1000, sampling_rate=audio.frame_rate)
            else:
                censor = pydub.AudioSegment.silent(duration=duration)
            modified_segments.append(censor)
            prior = end
            if i == len(silence_segments) - 1:  # if this is the last segment, append the remainder of audio
                modified_segments.append(audio[end:])
        output = sum(modified_segments)
    else:
        output = audio

    audio_save_path = path_to_audio_file.replace('.wav', '_censored.wav')
    output.export(audio_save_path, format='wav', codec=CODEC)
    return audio_save_path


def censor_video(path_to_video, beep=True):
    audio_file = extract_audio(path_to_video)
    censored_audio_file = censor_audio(audio_file, beep=beep)
    censored_video_file = overlay_audio(path_to_video=path_to_video, path_to_audio=censored_audio_file)
    return censored_video_file


def overlay_audio(path_to_video, path_to_audio):
    """ resulting video may have decreased audio quality.
    In testing, file of video with overlaid audio is smaller file size than audio file alone.
    Haven't investigated fully """

    print('Overlaying')
    print(f'\tVideo: {path_to_video}')
    print(f'\tAudio: {path_to_audio}')

    video = VideoFileClip(path_to_video, audio_fps=FRAMERATE)
    audio = AudioFileClip(path_to_audio)
    video = video.set_audio(audio)

    save_name = path_to_video.replace('.mp4', '_overlaid.mp4')
    print('Saving overlaid video:')
    print(f'\t{save_name}')
    video.write_videofile(save_name)
    return save_name


# run demo
if __name__ == '__main__':
    file = 'demo/demo.mp4'  # change to a .wav file to process a standalone audio file
    if file.endswith('.mp4'):
        censored_file = censor_video(file, beep=BEEP)
    elif file.endswith('.wav'):
        censored_file = censor_audio(file, beep=BEEP)
    else:
        raise ValueError(f"Input file must be .wav or .mp4. Unsupported type: {file}")
    print('Censored result file:', censored_file)

