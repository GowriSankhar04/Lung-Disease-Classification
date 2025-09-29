import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import entropy
from nolds import dfa
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import joblib

# Load models and preprocessing objects
rf_model = joblib.load("random_forest_model.pkl")
mlp_model = tf.keras.models.load_model("mlp_model.keras", custom_objects={
    "focal_loss_fn": lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(y_true, y_pred)
})
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("labelencoder.pkl")

# Constants
SAMPLE_RATE = 16000
DURATION = 20
SEGMENT_LENGTH = SAMPLE_RATE * DURATION
MFCC_FEATURES = 40
HOP_LENGTH = 128

# ------------------------ FEATURE EXTRACTION ------------------------
def extract_all_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < SEGMENT_LENGTH:
            y = np.pad(y, (0, SEGMENT_LENGTH - len(y)))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_FEATURES, hop_length=HOP_LENGTH)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        delta_mfcc = np.mean(librosa.feature.delta(mfcc), axis=1)

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH))

        stft = np.abs(librosa.stft(y, n_fft=256, hop_length=HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=256)
        envelope = np.abs(librosa.util.frame(y, frame_length=256, hop_length=HOP_LENGTH).mean(axis=0))
        envelope_smooth = librosa.util.normalize(np.convolve(envelope, np.ones(500)/500, mode='same'))

        peaks, _ = find_peaks(envelope, height=np.mean(envelope)*2, distance=50)
        crackle_count = len(peaks)

        fine_crackle_ratio = 0
        if crackle_count > 0:
            valid_peaks = peaks[peaks < stft.shape[1]]
            if len(valid_peaks) > 0:
                crackle_freqs = np.mean(stft[:, valid_peaks], axis=1)
                fine_crackle_ratio = np.sum(crackle_freqs[freqs > 400]) / np.sum(crackle_freqs) if np.sum(crackle_freqs) > 0 else 0

        high_freq_energy = np.mean(stft[(freqs > 400) & (freqs < 1000)], axis=0)
        wheeze_ratio = np.sum(high_freq_energy > np.mean(high_freq_energy)*1.5) / high_freq_energy.size

        breath_peaks, _ = find_peaks(envelope_smooth, distance=sr//2, height=np.mean(envelope_smooth))
        respiratory_rate = len(breath_peaks) * (60 / DURATION)

        ie_ratio = 1.0
        if len(breath_peaks) > 1:
            insp_peaks, _ = find_peaks(-envelope_smooth, distance=sr//2)
            cycle_durations = np.diff(breath_peaks) * (HOP_LENGTH / sr)
            exp_durations = np.diff(insp_peaks) * (HOP_LENGTH / sr) if len(insp_peaks) > 1 else cycle_durations
            ie_ratio = np.mean(cycle_durations) / np.mean(exp_durations) if len(exp_durations) > 0 else 1.0

        energy_low = np.mean(np.sum(stft[(freqs >= 100) & (freqs < 400)], axis=0)**2)
        energy_high = np.mean(np.sum(stft[(freqs >= 400) & (freqs < 1000)], axis=0)**2)

        tonal_energy = np.mean(stft[(freqs > 50) & (freqs < 200)], axis=0)
        ventilator_ratio = np.sum(tonal_energy > np.mean(tonal_energy)*2) / tonal_energy.size

        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=1000, sr=sr, hop_length=HOP_LENGTH)
        f0_mean = np.nanmean(f0) if np.any(voiced_flag) else 0
        f0_hi = np.nanmax(f0) if np.any(voiced_flag) else 0
        f0_lo = np.nanmin(f0) if np.any(voiced_flag) else 0

        jitter_abs = jitter_rap = jitter_ppq = jitter_ddp = 0
        if np.any(voiced_flag) and len(f0[~np.isnan(f0)]) > 1:
            f0_diff = np.diff(f0[~np.isnan(f0)])
            jitter_abs = np.mean(np.abs(f0_diff))
            jitter_rap = jitter_abs / f0_mean if f0_mean > 0 else 0
            jitter_ppq = np.mean(np.abs(f0_diff[:3])) / f0_mean if len(f0_diff) >= 3 and f0_mean > 0 else 0
            jitter_ddp = jitter_rap * 3

        peaks, _ = find_peaks(y, distance=50)
        shimmer = np.mean(np.abs(np.diff(np.abs(y[peaks])))) / np.mean(np.abs(y[peaks])) if len(peaks) > 1 else 0

        y_short = librosa.resample(y[:SAMPLE_RATE * 5], orig_sr=SAMPLE_RATE, target_sr=8000)
        hnr = np.mean(librosa.effects.harmonic(y_short)) / np.mean(librosa.effects.percussive(y_short)) if np.mean(librosa.effects.percussive(y_short)) > 0 else 0
        nhr = 1 / hnr if hnr > 0 else 0

        rpde = entropy(np.histogram(np.diff(peaks), bins=50, density=True)[0]) if len(peaks) > 1 else 0
        dfa_val = dfa(y_short) if len(y_short) > 100 else 0
        ppe = entropy(np.histogram(f0[~np.isnan(f0)], bins=50, density=True)[0]) if np.any(voiced_flag) else 0
        spread1 = f0_hi - f0_lo if np.any(voiced_flag) else 0
        spread2 = np.std(f0[~np.isnan(f0)]) if np.any(voiced_flag) else 0

        features = np.concatenate([
            mfcc_mean, mfcc_var, delta_mfcc,
            [spectral_centroid, zcr, spectral_rolloff, energy_low, energy_high,
             crackle_count, fine_crackle_ratio, wheeze_ratio, respiratory_rate,
             ie_ratio, ventilator_ratio,
             f0_mean, f0_hi, f0_lo, jitter_abs, jitter_rap, jitter_ppq, jitter_ddp,
             shimmer, hnr, nhr, rpde, dfa_val, ppe, spread1, spread2]
        ])
        return features
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return None
def extract_features(file_path, duration=5, sr=16000):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        y = librosa.util.fix_length(y, size=sr*duration)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.vstack([mfcc, delta, delta2])
        return np.mean(features, axis=1)  # shape: (60,)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
# ------------------------ PREDICTION ------------------------
def predict_audio(audio_path):
    x=predict_single_file(audio_path)
    if x==1:

        raw_features = extract_all_features(audio_path)
        if raw_features is None:
            return {"status": "error", "message": "Feature extraction failed."}

        feature_names = (
            [f'mfcc_mean_{i}' for i in range(40)] +
            [f'mfcc_var_{i}' for i in range(40)] +
            [f'delta_mfcc_{i}' for i in range(40)] +
            [
                'spectral_centroid', 'zero_crossing_rate', 'spectral_rolloff',
                'energy_low_band', 'energy_high_band',
                'crackle_count', 'fine_crackle_ratio', 'wheeze_ratio', 'respiratory_rate',
                'ie_ratio', 'ventilator_ratio',
                'f0_mean', 'f0_hi', 'f0_lo',
                'jitter_abs', 'jitter_rap', 'jitter_ppq', 'jitter_ddp',
                'shimmer', 'hnr', 'nhr', 'rpde', 'dfa', 'ppe', 'spread1', 'spread2'
            ]
        )

        feature_names = joblib.load("feature_columns.pkl")
        df = pd.DataFrame([raw_features], columns=feature_names)
        X_scaled_full = scaler.transform(df)

        selected_features = [
            'mfcc_mean_1', 'mfcc_mean_3', 'mfcc_mean_2', 'mfcc_mean_8', 'mfcc_mean_37',
            'mfcc_mean_7', 'spectral_centroid', 'mfcc_var_10', 'energy_low_band', 'mfcc_var_9',
            'mfcc_mean_4', 'zero_crossing_rate', 'mfcc_mean_0', 'mfcc_var_11', 'spectral_rolloff',
            'mfcc_mean_32', 'wheeze_ratio', 'energy_high_band', 'mfcc_mean_6', 'ventilator_ratio',
            'crackle_count', 'respiratory_rate'
        ]
        X_scaled_subset = X_scaled_full[:, [feature_names.index(f) for f in selected_features]]

        rf_pred = rf_model.predict(X_scaled_full)
        rf_label = label_encoder.inverse_transform(rf_pred)[0]
        rf_conf = rf_model.predict_proba(X_scaled_full)[0].max()

        mlp_out = mlp_model.predict(X_scaled_subset, verbose=0)[0]
        mlp_pred = np.argmax(mlp_out)
        mlp_label = label_encoder.inverse_transform([mlp_pred])[0]
        mlp_conf = float(np.max(mlp_out))

        weights = [0.7, 0.3]
        ensemble_vote = np.bincount([rf_pred[0], mlp_pred], weights=weights).argmax()
        ensemble_label = label_encoder.inverse_transform([ensemble_vote])[0]

        return {
            "status": "success",
            "RF Prediction": rf_label,
            "RF Confidence": float(rf_conf),
            "MLP Prediction": mlp_label,
            "MLP Confidence": mlp_conf,
            "Ensemble Prediction": ensemble_label
        }
    else:
        return "⚠️ This doesn’t seem to be a lung sound. Please upload a proper lung recording."

def predict_single_file(file_path):
    """
    Extracts features from a single audio file, scales them,
    and predicts the class using the loaded model.
    """
    model= joblib.load("lung_vs_nonlung.pkl")
    scaler_one= joblib.load("lung_vs_nonlung_scaler.pkl")
    # Extract features
    features = extract_features(file_path)
    if features is None:
        return "Could not process the audio file."

    # Reshape and scale the features
    features = features.reshape(1, -1)
    features_scaled = scaler_one.transform(features)

    # Make a prediction
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)

    # Return the result
    if prediction[0] == 1:
        return 1
    else:
        return 0

