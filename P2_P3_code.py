import parselmouth
from parselmouth.praat import call
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import freqz, lfilter, convolve
from scipy.io import wavfile

# ==========================================
# 1. SETUP AND CONFIGURATION
# ==========================================
audio_path = r"C:\Users\medha\Desktop\EE-679\Assignment_1\P1_part_a.wav"
textgrid_path = r"C:\Users\medha\Desktop\EE-679\Assignment_1\P1_part_a.TextGrid"

# Vowels to visualize in Scatter Plot
VOWELS_ALL = ["a", "æ", "ə", "i", "ɪ", "u", "ʊ", "e", "o", "ɑ", "ɔ", "ɛ", "ʌ"]

# Vowels specifically required for Generation (P3)
# We will try to find these specific labels in your TextGrid.
# If your TextGrid uses IPA 'ɑ' for 'a', we handle that mapping below.
TARGET_VOWELS_GEN = ['i', 'u', 'a', 'ɑ'] 

snd = parselmouth.Sound(audio_path)
Fs = 8000  # Sampling rate for generation (Part P3 requirement)

# ==========================================
# 2. (a) PITCH ANALYSIS (P2)
# ==========================================
print("--- Part (a): Pitch Analysis ---")

pitch = snd.to_pitch()
pitch_values = pitch.selected_array["frequency"]
pitch_values_voiced = pitch_values[pitch_values > 0]

avg_f0 = np.mean(pitch_values_voiced)
print(f"Average F0: {avg_f0:.2f} Hz")

plt.figure(figsize=(10, 6))
plt.hist(pitch_values_voiced, bins=40, color='skyblue', edgecolor='black')
plt.xlabel("F0 (Hz)")
plt.ylabel("Count")
plt.title(f"Histogram of F0 (Avg: {avg_f0:.1f} Hz)")
plt.grid(axis='y', alpha=0.75)
plt.savefig("part_a_pitch_histogram.png")
# plt.show() # Commented out to let script run smoothly to end

# ==========================================
# 3. (b) FORMANT ANALYSIS & DATA GATHERING
# ==========================================
print("\n--- Part (b): Formant Analysis & Param Extraction ---")

formant = snd.to_formant_burg()
tg = parselmouth.read(textgrid_path)
tier_index = 2
num_intervals = call(tg, "Get number of intervals", tier_index)

# Data containers
f1_list, f2_list, labels_list = [], [], []

# Storage for Generation Params: {'i': {'f1':[], 'f2':[], 'f3':[]}, ...}
gen_data = {v: {'f1': [], 'f2': [], 'f3': []} for v in TARGET_VOWELS_GEN}

print(f"Processing {num_intervals} intervals...")

for i in range(1, num_intervals + 1):
    label = call(tg, "Get label of interval", tier_index, i).strip()

    if label in VOWELS_ALL:
        start = call(tg, "Get start time of interval", tier_index, i)
        end = call(tg, "Get end time of interval", tier_index, i)
        
        # Get Means (Prompt requires Average over segment)
        f1 = call(formant, "Get mean", 1, start, end, "Hertz")
        f2 = call(formant, "Get mean", 2, start, end, "Hertz")
        f3 = call(formant, "Get mean", 3, start, end, "Hertz") # Added for P3

        if not np.isnan(f1) and not np.isnan(f2):
            # For Scatter Plot
            f1_list.append(f1)
            f2_list.append(f2)
            labels_list.append(label)
            
            # For Vowel Generation (collect stats if it matches target)
            if label in gen_data:
                if not np.isnan(f3): # Ensure F3 is valid
                    gen_data[label]['f1'].append(f1)
                    gen_data[label]['f2'].append(f2)
                    gen_data[label]['f3'].append(f3)

# Scatter Plot
plt.figure(figsize=(10, 8))
plt.scatter(f2_list, f1_list, c='red', alpha=0.3, s=50)
for i, label in enumerate(labels_list):
    plt.text(f2_list[i], f1_list[i], label, fontsize=12, ha='center', va='center')
plt.xlabel("F2 (Hz)")
plt.ylabel("F1 (Hz)")
plt.title("Vowel Formant Scatter Plot")
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("part_b_vowel_scatter.png")
print("Saved scatter plot.")

# ==========================================
# 4. (P3) VOWEL GENERATION LOGIC
# ==========================================
print("\n--- Part (P3): Vowel Generation ---")

# --- A. Define Filter and Helper Functions ---

def design_vowel_filter(formants, Fs):
    """Creates cascade of 3 second-order sections."""
    Q = random.uniform(5, 10) # Random Q as requested
    a_total = np.array([1.0])
    params_log = []

    for Fi in formants:
        Bi = Fi / Q
        theta = 2 * np.pi * Fi / Fs
        r = np.exp(-2 * np.pi * Bi / Fs)
        
        # H(z) denominator: 1 - 2r*cos(theta)z^-1 + r^2*z^-2
        a_section = np.array([1, -2 * r * np.cos(theta), r**2])
        a_total = convolve(a_total, a_section)
        params_log.append((Fi, Bi))
        
    return a_total, Q, params_log

def plot_response(a_coeffs, vowel, params):
    w, h = freqz(1, a_coeffs, worN=2048, fs=Fs)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    
    plt.figure(figsize=(8, 4))
    plt.plot(w, mag_db)
    plt.title(f"Frequency Response for /{vowel}/")
    plt.xlabel("Freq (Hz)")
    plt.ylabel("Mag (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"P3_resp_{vowel}.png")
    plt.close()

# --- B. Prepare Parameters from Analysis ---

# Default fallbacks if a vowel wasn't found in your audio
final_params = {
    'i': [270, 2290, 3010],
    'u': [300, 870, 2240],
    'a': [730, 1090, 2440]
}

# Overwrite defaults with YOUR measured averages
for v in ['i', 'u', 'a']:
    # Check if we have data for 'a' or 'ɑ'
    key = v
    if v == 'a' and len(gen_data['a']['f1']) == 0 and len(gen_data['ɑ']['f1']) > 0:
        key = 'ɑ' # Use 'ɑ' data for 'a' target
        
    if len(gen_data[key]['f1']) > 0:
        avg_f1 = np.mean(gen_data[key]['f1'])
        avg_f2 = np.mean(gen_data[key]['f2'])
        avg_f3 = np.mean(gen_data[key]['f3'])
        final_params[v] = [avg_f1, avg_f2, avg_f3]
        print(f"Using measured params for /{v}/: F1={avg_f1:.0f}, F2={avg_f2:.0f}, F3={avg_f3:.0f}")
    else:
        print(f"Warning: /{v}/ not found in recording. Using standard fallback values.")

# --- C. Generate Signals ---

DURATION = 1.0
t = np.arange(0, DURATION, 1/Fs)

# 1. Impulse Train (Source)
impulse_train = np.zeros_like(t)
period_samples = int(Fs / avg_f0)
impulse_train[0::period_samples] = 1.0

# 2. Rectified Cosine (Bonus c Source)
cosine_wave = np.cos(2 * np.pi * avg_f0 * t)
rect_cosine = np.maximum(0, cosine_wave)

# --- D. Synthesis Loop ---

for v_name in ['i', 'u', 'a']:
    print(f"Synthesizing /{v_name}/...")
    formants = final_params[v_name]
    
    # Design Filter
    a_coeffs, Q, p_log = design_vowel_filter(formants, Fs)
    
    # Plot Response (Part a)
    plot_response(a_coeffs, v_name, p_log)
    
    # Filter Impulse Train (Part b)
    sig_imp = lfilter([1], a_coeffs, impulse_train)
    sig_imp = sig_imp / np.max(np.abs(sig_imp)) # Normalize
    wavfile.write(f"P3_{v_name}_impulse.wav", Fs, (sig_imp * 32767).astype(np.int16))
    
    # Filter Rectified Cosine (Part c - Bonus)
    sig_cos = lfilter([1], a_coeffs, rect_cosine)
    sig_cos = sig_cos / np.max(np.abs(sig_cos)) # Normalize
    wavfile.write(f"P3_{v_name}_cosine.wav", Fs, (sig_cos * 32767).astype(np.int16))

print("\nProcessing Complete.")
print("Outputs: 3 Response plots, 6 WAV files (Impulse & Cosine versions for i, u, a)")