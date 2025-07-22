import numpy as np
import librosa
import torch
import torchaudio
from IPython.display import Audio
from pprint import pprint
import torch
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


audio, sr = librosa.load("E:\\Daicwoz\\extracted_wav_files\\301_AUDIO.wav", sr=16000)

print(audio.shape)


model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)



(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


SAMPLING_RATE=sr
speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=SAMPLING_RATE)
print(speech_timestamps)

# save_audio('only_speech.wav',
#            collect_chunks(speech_timestamps, audio), sampling_rate=SAMPLING_RATE) 


#  Plot the audio waveform
plt.figure(figsize=(14, 8))
librosa.display.waveshow(audio, sr=sr)

# Add vertical lines for start and stop times
for timestamp in speech_timestamps:
    start = timestamp['start'] / sr
    end = timestamp['end'] / sr
    plt.axvline(x=start, color='g', linestyle='--', label='Start')
    plt.axvline(x=end, color='r', linestyle='--', label='End')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform with Speech Segments')

# To avoid duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# Show the plot
plt.show()