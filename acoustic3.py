import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize

# keystroke file 
loud_then_quiet = AudioSegment.from_file("Sample_1.wav")
normalize_then_quiet = loud_then_quiet.normalize()
normalize_then_quiet.export("Sample_1Normal.wav", format="wav")
# Load the keystroke file
audio_data, sample_rate = librosa.load("Sample_1Normal.wav", sr=None)

# Generating the spectrogram
hop_length = 512
n_fft = 2048  # Adjust the number of FFT points
fmax = 10000   # Maximum frequency value

spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128, n_fft=n_fft, 
                                             hop_length=hop_length, fmax=fmax)
log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

# threshold to see where there are values more than -25 decibels
threshold = -20

# Analyze the spectrogram for events where its louder than -20 decibels, indicating a space bar
space_bar_events = np.where(log_spectrogram > threshold)

# Convert time indices to actual time values
time_values = space_bar_events[1] * hop_length / sample_rate

# Convert frequency indices to actual frequency values
freq_values = librosa.core.mel_frequencies(n_mels=128, fmin=0.0, fmax=fmax)

num_spaceBar = 0
time_ind = []
# Print detected events with actual time and frequency values
print("Detected louder than -20 decibel events:")
for event, time_value in zip(zip(space_bar_events[0], space_bar_events[1]), time_values):
    freq_value = freq_values[event[0]]
    print(f"Time: {time_value:.2f}s - Frequency: {freq_value:.2f} Hz")
    if(freq_value > 2000):
        time_ind.append(time_value)
time_ind.sort()
time_num = 0
time_count = 0
visited_time = []
# Depending on where the sounds occured most, those time values are going to be grouped
def valExists(num, valArray):
    for values in valArray:
        if num < values + .10 and num > values - .10:
            return True
    return False
for i in range(len(time_ind)):
    print(time_ind[i])
    if(time_num < time_ind[i] + 0.10 and time_num > time_ind[i] - 0.10):
        time_count += 1
        if(time_count >= 15 and not valExists(time_ind[i],visited_time)):
            num_spaceBar += 1
            visited_time.append(time_ind[i])
    else:
        time_count = 1
        time_num = time_ind[i]
print("Num of words typed: " + str(num_spaceBar + 1))
# Display the spectrogram
librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram with Space Bar Events')
plt.show()