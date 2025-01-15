import openai
from faster_whisper import WhisperModel
import torchaudio
from TTS.api import TTS
import sounddevice as sd
import wave
import joblib
import numpy as np
import librosa
import soundfile

# OpenAI API key
OPENAI_API_KEY = "OpenAI API key"


def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            mel = np.mean(mel_spectrogram.T, axis=0)
            result = np.hstack((result, mel))
    return result

model = joblib.load('./model/saved_model.pkl')


def predict_emotion(audio_file):
    feature = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1, -1)
    prediction = model.predict(feature)
    return prediction[0]


def analyze_transcript(transcript):
    openai.api_key = OPENAI_API_KEY
    
    prompt = (f"Based on the transcript: '{transcript}', "
              f"provide a concise response to the user's query.")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.5,
            top_p=1.0,
            n=1
        )

        message_content = response['choices'][0]['message']['content'].strip()
        return message_content

    except Exception as e:
        print(f"Error analyzing transcript: {e}")
        return {"error": str(e)}


def text_to_speech(text, file_path):
    tts_model = TTS(model_name='tts_models/en/ljspeech/vits--neon')
    tts_model.tts_to_file(text=text, file_path=file_path)
    return file_path


def play_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.numpy().T
    sd.play(waveform, sample_rate)
    sd.wait()


def record_audio(duration, filename):
    fs = 16000  # Sample rate
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    print("Recording finished.")
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) 
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())


def run_speech_to_speech():
    duration = 10  # seconds
    input_audio_file = "./audio/real_time_input_for_speech_to_speech_model.wav"
    output_audio_file = "./audio/response_output_for_speech_to_speech_model.wav"

    record_audio(duration, input_audio_file)

    predicted_emotion = predict_emotion(input_audio_file)
    print(f"Predicted Emotion: {predicted_emotion}")

    model_size = "large-v2"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(input_audio_file, beam_size=5)
    print(f"Detected language '{info.language}' with probability {info.language_probability}")

    transcription = " ".join([segment.text for segment in segments])
    print("Transcription:", transcription)

    concise_response = analyze_transcript(transcription)
    print("Concise Response:", concise_response)

    response_audio_path = text_to_speech(concise_response, output_audio_file)
    print(f"Generated speech saved to {response_audio_path}")

    play_audio(response_audio_path)
