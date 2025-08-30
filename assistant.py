import os
import wave
import requests
import numpy as np
import torch
import torchaudio
import pyaudio
import pyttsx3
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_NAME = "MODEL_NAME"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def speak(text: str) -> None:
  print(text)
  engine = pyttsx3.init()
  engine.say(text)
  engine.runAndWait()

# speak("Привет! Я - ассистент. Чем могу помочь?")

def record_audio(filename: str, duration: int = 5, sample_rate: int = 16000) -> None:
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
  print("start recording...")
  frames = [stream.read(1024) for _ in range(int(sample_rate / 1024 * duration))]
  print("end recording.")
  stream.stop_stream()
  stream.close()
  p.terminate()

  with wave.open(filename, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))

# record_audio("command.wav")

def transcribe_audio(filename: str) -> str:
    try:
        with wave.open(filename, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except (wave.Error, FileNotFoundError):
        return ""

    if n_channels != 1 or sampwidth != 2:
        return ""

    # Преобразуем в float32 [-1, 1]
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Проводим ресемплинг до 16 кГц, если нужно
    if sr != 16000:
        tensor = torch.from_numpy(audio).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=16000)
        audio = tensor.squeeze(0).numpy()
        sr = 16000

    # Запускаем модель для получения текста
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE) if "attention_mask" in inputs else None
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0].strip()

# text = transcribe_audio("command.wav")
# speak(text)