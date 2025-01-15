import openai
import torchaudio
from TTS.api import TTS
import sounddevice as sd

# OpenAI API key
OPENAI_API_KEY = "OpenAI API key"

# Initialize TTS model
tts_model = TTS(model_name='tts_models/en/ljspeech/vits--neon')

def get_gpt_response(user_input):
    openai.api_key = OPENAI_API_KEY
    
    prompt = (f"User said: '{user_input}'. Respond in a short and concise manner.")
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.5,
            top_p=1.0,
            n=1
        )

        message_content = response['choices'][0]['message']['content'].strip()
        return message_content

    except Exception as e:
        print(f"Error getting GPT response: {e}")
        return "Sorry, I couldn't process your request."

def text_to_speech(text, file_path):
    tts_model.tts_to_file(text=text, file_path=file_path)
    return file_path

def play_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.numpy().T
    sd.play(waveform, sample_rate)
    sd.wait()

def run_text_to_speech():
    user_input = input("Please enter your query: ")
    gpt_response = get_gpt_response(user_input)
    print(f"GPT Response: {gpt_response}")
    
    audio_path = "./audio/output_GPT.wav"
    text_to_speech(gpt_response, audio_path)
    play_audio(audio_path)
