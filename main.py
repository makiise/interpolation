# Install required packages if not already installed
# Run these commands in your terminal or command prompt:
# pip install gtts librosa matplotlib playsound scipy soundfile

from gtts import gTTS
import librosa
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound
from scipy.interpolate import interp1d, CubicSpline
import soundfile as sf

# Convert text to speech
text = "me var arti tsida, me sitsili minda, ha ha ha"
tts = gTTS(text)
tts.save("speech.mp3")

# Play the original audio file
print("Playing original audio...")
playsound("speech.mp3")

# Load audio file
y, sr = librosa.load("speech.mp3", sr=None)

# Interpolation functions
def interpolate_signal(y, method='linear'):
    x = np.arange(len(y))
    x_new = np.linspace(0, len(y) - 1, len(y) * 2)  # Double the length for interpolation

    if method == 'linear':
        interpolator = interp1d(x, y, kind='linear')
    elif method == 'cubic':
        interpolator = CubicSpline(x, y)
    elif method == 'nearest':
        interpolator = interp1d(x, y, kind='nearest')
    else:
        raise ValueError("Unsupported interpolation method")

    y_new = interpolator(x_new)
    return y_new

# Store interpolated signals
interpolated_signals = {}
methods = ['linear', 'cubic', 'nearest']

for method in methods:
    y_interpolated = interpolate_signal(y, method)
    interpolated_signals[method] = y_interpolated
    output_file = f"speech_{method}.wav"
    sf.write(output_file, y_interpolated, sr)
    print(f"Playing {method} interpolated audio...")
    playsound(output_file)

# Ensure the interpolated signals are different by comparing a few samples
for method in methods:
    print(f"{method.capitalize()} interpolation: first 10 samples: {interpolated_signals[method][:10]}")

# Plot the waveform of the original and interpolated signals for comparison
plt.figure(figsize=(12, 8))

# Find the global minimum and maximum values across all signals
all_signals = [y] + list(interpolated_signals.values())
global_min = min([np.min(signal) for signal in all_signals])
global_max = max([np.max(signal) for signal in all_signals])

# Plot all signals on the same subplot with shared y-axis
for i, method in enumerate(['Original'] + methods):
    if method == 'Original':
        signal = y
    else:
        signal = interpolated_signals[method.lower()]
    plt.plot(signal, label=f'{method}')

plt.title('Original and Interpolated Audio Waveforms')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.ylim(global_min, global_max)  # Set the same y-axis limits for all signals
plt.legend()

plt.tight_layout()
plt.show()