import parselmouth
from parselmouth.praat import call
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import freqz, lfilter, convolve
from scipy.io import wavfile

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
# UPDATE THESE PATHS TO YOUR ACTUAL FILE LOCATIONS
audio_path = r"C:\Users\medha\Desktop\EE-679\Assignment_1\P1_part_a.wav"
textgrid_path = r"C:\Users\medha\Desktop\EE-679\Assignment_1\P1_part_a.TextGrid"

# Sampling rate for Generation (P3 requires Fs=8000)
Fs_gen = 8000 

# Vowels to analyze for Scatter Plot (P2)
VOWELS_ALL = ["a", "æ", "ə", "i", "ɪ", "u", "ʊ", "e", "o", "ɑ", "ɔ", "ɛ", "ʌ"]

# Vowels required for Generation (P3)
# Note: 'ɑ' is often used for 'a' in TextGrids; we treat them as the same target
TARGET_VOWELS_GEN = ['i', 'u', 'a', 'ɑ']

# Load Parselmouth Objects
snd = parselmouth.Sound(audio_path)
tg = parselmouth.read(textgrid_path)

# ==========================================
# PART P2: ACOUSTIC ANALYSIS
# ==========================================
print("--- STARTING PART P2: ACOUSTIC ANALYSIS ---")

# --- P2(a) PITCH ANALYSIS ---
pitch = snd.to_pitch()
pitch_values = pitch.selected_array["frequency"]
pitch_values_voiced = pitch_values[pitch_values > 0] # Filter unvoiced

# 1. Calculate Average F0
avg_f0 = np.mean(pitch_values_voiced)
print(f"Average F0 (from P2a): {avg_f0:.2f} Hz")

# 2. Plot Histogram
plt.figure(figsize=(10, 6))
plt.hist(pitch_values_voiced, bins=40, color='skyblue', edgecolor='black')
plt.xlabel("F0 (Hz)")
plt.ylabel("Count")
plt.title(f"P2(a): Histogram of Pitch F0 (Avg: {avg_f0:.1f} Hz)")
plt.grid(axis='y', alpha=0.75)
plt.savefig("P2a_pitch_histogram.png")
plt.close()
print("Saved: P2a_pitch_histogram.png")

# --- P2(b) FORMANT ANALYSIS ---
formant = snd.to_formant_burg()
tier_index = 2 # Usually Phoneme tier
num_intervals = call(tg, "Get number of intervals", tier_index)

# Lists for Scatter Plot
f1_scatter, f2_scatter, label_scatter = [], [], []

# Dictionary to store all instances of specific vowels for P3 averaging
# Structure: {'i': {'f1': [], 'f2': [], 'f3': []}, ...}
vowel_data_collection = {v: {'f1': [], 'f2': [], 'f3': []} for v in TARGET_VOWELS_GEN}

print(f"Analyzing {num_intervals} intervals in TextGrid...")

for i in range(1, num_intervals + 1):
    label = call(tg, "Get label of interval", tier_index, i).strip()

    if label in VOWELS_ALL:
        start = call(tg, "Get start time of interval", tier_index, i)
        end = call(tg, "Get end time of interval", tier_index, i)
        
        # EXTRACT MEAN FORMANTS (as required by prompt)
        f1 = call(formant, "Get mean", 1, start, end, "Hertz")
        f2 = call(formant, "Get mean", 2, start, end, "Hertz")
        f3 = call(formant, "Get mean", 3, start, end, "Hertz") # Needed for P3

        if not np.isnan(f1) and not np.isnan(f2):
            # Store for P2 scatter plot
            f1_scatter.append(f1)
            f2_scatter.append(f2)
            label_scatter.append(label)
            
            # Store for P3 generation if it matches target vowels
            if label in vowel_data_collection:
                if not np.isnan(f3):
                    vowel_data_collection[label]['f1'].append(f1)
                    vowel_data_collection[label]['f2'].append(f2)
                    vowel_data_collection[label]['f3'].append(f3)

# 3. Plot Scatter (Vowel Triangle)
plt.figure(figsize=(10, 8))
plt.scatter(f2_scatter, f1_scatter, c='red', alpha=0.3, s=50)

# Annotate points with text
for i, txt in enumerate(label_scatter):
    plt.text(f2_scatter[i], f1_scatter[i], txt, fontsize=12, ha='center', va='center')

plt.xlabel("F2 (Hz)")
plt.ylabel("F1 (Hz)")
plt.title("P2(b): Vowel Formant Scatter Plot (Vowel Triangle)")
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("P2b_vowel_scatter.png")
plt.close()
print("Saved: P2b_vowel_scatter.png")


# ==========================================
# PART P3: VOWEL GENERATION
# ==========================================
print("\n--- STARTING PART P3: VOWEL GENERATION ---")

# --- 1. CALCULATE PARAMETERS FROM DATA ---
# Default standard male values (fallback if vowel not found in audio)
generation_params = {
    'i': [270, 2290, 3010],
    'u': [300, 870, 2240],
    'a': [730, 1090, 2440]
}

# Update with YOUR actual measured averages
print("Calculating average formants from your recording...")
for v_target in ['i', 'u', 'a']:
    # Handle 'a' vs 'ɑ' mapping
    keys_to_check = [v_target]
    if v_target == 'a': keys_to_check.append('ɑ')
    
    found_data = False
    for key in keys_to_check:
        if len(vowel_data_collection[key]['f1']) > 0:
            avg_f1 = np.mean(vowel_data_collection[key]['f1'])
            avg_f2 = np.mean(vowel_data_collection[key]['f2'])
            avg_f3 = np.mean(vowel_data_collection[key]['f3'])
            generation_params[v_target] = [avg_f1, avg_f2, avg_f3]
            print(f"  /{v_target}/ using measured: F1={avg_f1:.0f}, F2={avg_f2:.0f}, F3={avg_f3:.0f}")
            found_data = True
            break
    
    if not found_data:
        print(f"  /{v_target}/ not found in recording. Using standard fallback.")


# --- 2. FILTER DESIGN & SIGNAL GENERATION FUNCTIONS ---

def design_vowel_filter(formants, fs):
    """
    Constructs H(z) = H1(z)H2(z)H3(z)
    Returns coefficients and parameters used.
    """
    Q = random.uniform(5, 10) # Random Q in [5, 10]
    
    a_total = np.array([1.0])
    params_info = []

    for Fi in formants:
        Bi = Fi / Q
        
        # Prompt Formulas:
        # theta = 2*pi*Fi / Fs
        # r = exp(-2*pi*Bi / Fs)
        theta = 2 * np.pi * Fi / fs
        r = np.exp(-2 * np.pi * Bi / fs)
        
        # Denominator (a) for 2nd order section:
        # 1 - 2r*cos(theta)z^-1 + r^2*z^-2
        a_section = np.array([1, -2 * r * np.cos(theta), r**2])
        
        # Cascade filters (convolution in time domain)
        a_total = convolve(a_total, a_section)
        
        params_info.append((Fi, Bi))

    return a_total, Q, params_info

def plot_freq_response(a_coeffs, vowel_name, params, Q):
    """Plots 10*log10|H(w)|^2 with values annotated."""
    w, h = freqz(1, a_coeffs, worN=2048, fs=Fs_gen)
    
    # Magnitude Squared in dB: 10*log10(|H|^2) = 20*log10(|H|)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    
    plt.figure(figsize=(10, 6))
    plt.plot(w, mag_db, linewidth=2)
    
    # Annotation Text
    info_text = f"Vowel /{vowel_name}/\nQ Factor: {Q:.2f}\n"
    for i, (f, b) in enumerate(params):
        info_text += f"F{i+1}: {f:.1f} Hz, B{i+1}: {b:.1f} Hz\n"
        # Draw line at formant
        plt.axvline(x=f, color='r', linestyle='--', alpha=0.4)
        plt.text(f, np.max(mag_db)-10, f" F{i+1}", color='r', rotation=90)

    # Text box in corner
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.9))

    plt.title(f"P3(a): Frequency Response |H(w)|^2 for /{vowel_name}/")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, 4000)
    
    filename = f"P3_response_{vowel_name}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

def generate_excitations(fs, f0, duration):
    t = np.arange(0, duration, 1/fs)
    
    # 1. Impulse Train
    impulse = np.zeros_like(t)
    period_samples = int(fs / f0)
    impulse[0::period_samples] = 1.0
    
    # 2. Rectified Cosine (Bonus)
    cosine = np.cos(2 * np.pi * f0 * t)
    rect_cosine = np.maximum(0, cosine)
    
    return impulse, rect_cosine

# --- 3. MAIN GENERATION LOOP ---

DURATION = 1.0
impulse_src, cosine_src = generate_excitations(Fs_gen, avg_f0, DURATION)

print(f"\nSynthesizing vowels at F0 = {avg_f0:.2f} Hz...")

for vowel in ['i', 'u', 'a']:
    formants = generation_params[vowel]
    
    # A. Design Filter
    a_coeffs, Q, params = design_vowel_filter(formants, Fs_gen)
    
    # B. Plot Response (P3a)
    plot_freq_response(a_coeffs, vowel, params, Q)
    
    # C. Generate Impulse Audio (P3b)
    wav_imp = lfilter([1.0], a_coeffs, impulse_src)
    wav_imp = wav_imp / np.max(np.abs(wav_imp)) # Normalize
    fname_imp = f"P3_vowel_{vowel}_impulse.wav"
    wavfile.write(fname_imp, Fs_gen, (wav_imp * 32767).astype(np.int16))
    
    # D. Generate Cosine Audio (P3c Bonus)
    wav_cos = lfilter([1.0], a_coeffs, cosine_src)
    wav_cos = wav_cos / np.max(np.abs(wav_cos)) # Normalize
    fname_cos = f"P3_vowel_{vowel}_cosine.wav"
    wavfile.write(fname_cos, Fs_gen, (wav_cos * 32767).astype(np.int16))
    
    print(f"Generated WAVs for /{vowel}/")