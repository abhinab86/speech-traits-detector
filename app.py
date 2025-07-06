from flask import Flask, render_template, request
import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
gender_model = joblib.load('gender_lightgbm_model.pkl')
emotion_model = load_model('Best_CNN_Model.keras')  # make sure this path is correct 

# Emotion label mapping
emotion_map = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}

# Audio processing parameters
sr = 16000        # Sample rate (ensure this matches training)
duration = 2      # Duration in seconds (used during training)
n_mels = 128      # Mel bands (used during training)

# Gender feature extraction
def extract_gender_features(audio_path):
    y, sr_local = librosa.load(audio_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr_local, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.mean(f0)
    except:
        f0_mean = 0

    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_local))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr_local))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr_local))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    features = np.hstack([mfccs_mean, f0_mean, spec_centroid, spec_bw, rolloff, zcr])
    return features.reshape(1, -1)

# Emotion feature extraction (Mel spectrogram)
def extract_mel(y_audio):
    max_len = sr * duration
    if len(y_audio) < max_len:
        y_audio = np.pad(y_audio, (0, max_len - len(y_audio)))
    else:
        y_audio = y_audio[:max_len]

    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    # Ensure width is 130 (time steps)
    if mel_db.shape[1] < 130:
        pad_width = 130 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :130]

    return mel_db

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return "No file uploaded", 400

    audio = request.files['audio_file']
    if audio.filename == '':
        return "No selected file", 400

    file_path = os.path.join('temp_audio', audio.filename)
    os.makedirs('temp_audio', exist_ok=True)
    audio.save(file_path)

    try:
        # Predict gender
        gender_features = extract_gender_features(file_path)
        gender_pred = gender_model.predict(gender_features)[0]
        gender_label = "Male" if gender_pred == 1 else "Female"

        # Predict emotion
        y_audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        mel_spec = extract_mel(y_audio)
        mel_spec = np.expand_dims(mel_spec, axis=-1)  # (128, ?, 1)
        mel_spec = np.expand_dims(mel_spec, axis=0)   # (1, 128, ?, 1)

        emotion_probs = emotion_model.predict(mel_spec)
        emotion_label = np.argmax(emotion_probs)
        emotion = emotion_map.get(emotion_label, "Unknown")

        # Placeholder for age (optional — you can integrate your age model similarly)


        age_group = "26-35"

    except Exception as e:
        return f"Error during prediction: {e}", 500
    finally:
        os.remove(file_path)

    return render_template('index.html',
                           gender=gender_label,
                           age=age_group,
                           emotion=emotion,
                           file_uploaded=True)

if __name__ == '__main__':
    app.run(debug=True)