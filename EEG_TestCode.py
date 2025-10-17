# ======================================================
#   EEG / Behavior Simulation + Hidden States + FFT + Spectrogram
#   (No external data or hmmlearn required)
# ======================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

np.random.seed(7)

# ------------------------------------------------------
# 1. Simulate hidden states and observable behavior
# ------------------------------------------------------
hidden_states = ["Hungry", "Full", "Awake", "Asleep"]
observations = ["Eating", "Walking", "Still", "Sleeping"]

T = 600  # total time steps
A = np.array([
    [0.7, 0.2, 0.1, 0.0],   # Hungry
    [0.2, 0.6, 0.2, 0.0],   # Full
    [0.1, 0.1, 0.7, 0.1],   # Awake
    [0.0, 0.0, 0.2, 0.8]    # Asleep
])
B = np.array([
    [0.6, 0.3, 0.1, 0.0],   # Hungry
    [0.2, 0.5, 0.3, 0.0],   # Full
    [0.0, 0.6, 0.3, 0.1],   # Awake
    [0.0, 0.0, 0.2, 0.8]    # Asleep
])

def simulate(A, B, start=0, T=600):
    nS, nO = B.shape
    states, obs = [start], [np.random.choice(nO, p=B[start])]
    for _ in range(T-1):
        s_prev = states[-1]
        s_new = np.random.choice(range(nS), p=A[s_prev])
        o_new = np.random.choice(range(nO), p=B[s_new])
        states.append(s_new); obs.append(o_new)
    return np.array(states), np.array(obs)

states, obs = simulate(A, B, start=2, T=T)

plt.figure(figsize=(12,3))
plt.plot(states, lw=2)
plt.yticks(range(len(hidden_states)), hidden_states)
plt.title("Simulated Hidden State Sequence")
plt.xlabel("Time step")
plt.tight_layout()
plt.savefig("hidden_states.png")
plt.show()

# ------------------------------------------------------
# 2. Simulate multichannel EEG-like signals
# ------------------------------------------------------
fs = 100  # Hz
t = np.arange(0, T/fs, 1/fs)

# Channel components: alpha (10 Hz), beta (20 Hz), noise
alpha = np.sin(2*np.pi*10*t)
beta  = 0.5*np.sin(2*np.pi*20*t)
noise = 0.3*np.random.randn(len(t))
signal = alpha + beta + noise

# Add modulation based on hidden state (less energy when asleep)
mask = (states == hidden_states.index("Asleep")).astype(float)
signal = signal * (1 - 0.5*mask[:len(signal)])  # reduce amplitude while asleep

plt.figure(figsize=(10,3))
plt.plot(t, signal)
plt.title("Synthetic EEG / Behavioral Signal (modulated by state)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("signal_time.png")
plt.show()

# ------------------------------------------------------
# 3. FFT Analysis
# ------------------------------------------------------
fft_vals = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(len(signal), 1/fs)
power = np.abs(fft_vals)**2

plt.figure(figsize=(8,4))
plt.plot(fft_freqs[:len(power)//2], power[:len(power)//2])
plt.title("Power Spectrum of Synthetic EEG")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.grid(True)
plt.tight_layout()
plt.savefig("power_spectrum.png")
plt.show()

# ------------------------------------------------------
# 4. Spectrogram (time–frequency)
# ------------------------------------------------------
f, t_spec, Sxx = spectrogram(signal, fs=fs, nperseg=256, noverlap=128)
plt.figure(figsize=(10,4))
plt.pcolormesh(t_spec, f, 10*np.log10(Sxx), shading='gouraud', cmap='magma')
plt.title("Spectrogram (Time–Frequency Energy)")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.savefig("spectrogram.png")
plt.show()

# ------------------------------------------------------
# 5. Summary and talking points
# ------------------------------------------------------
print("=== Summary ===")
print("Hidden states:", hidden_states)
print("Observations :", observations)
print("\nFFT peaks around 10 Hz (alpha) and 20 Hz (beta) bands,")
print("representing typical EEG frequency content during awake periods.")
print("Spectrogram shows reduced amplitude in these bands when 'Asleep'.")
print("\nNext step: replace synthetic signal with actual EEG recordings for comparison.")
