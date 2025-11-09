# imports
import base64, io, json, os, uuid, subprocess, sys, wave
from pathlib import Path

import pandas as pd
import os
import numpy as np

# for speech-to-text
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import soundfile as sf
import ffmpeg

# train test split
from sklearn.model_selection import train_test_split

# to read in json file
import json

# spacy
import spacy
from spacy.tokens import DocBin, Doc, Span
from tqdm import tqdm
from spacy.util import filter_spans
from spacy.scorer import Scorer
from spacy.training import Example

# for visualisation
from spacy import displacy

# Load model
nlp_trained = spacy.load('./output/model-best') # load the best model



# Setting up text-to-speech
MODEL_DIR = Path("models/vosk-model-small-en-us-0.15")  # change if you downloaded a different model
SAMPLE_RATE = 16000  # preferred for Vosk
MODEL = None  # lazy-load

def _ensure_model():
  global MODEL
  if MODEL is None:
    if not MODEL_DIR.exists():
      raise FileNotFoundError(
          f"Vosk model not found at {MODEL_DIR}. Update MODEL_DIR or re-run the setup cell.")
    print(f"Loading Vosk model from: {MODEL_DIR} ...")
    MODEL = Model(str(MODEL_DIR))
    print("Model ready.")

def record_wav(seconds=5, out_wav="take1.wav", sr=SAMPLE_RATE):
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    print(f"🎙️ Recording {seconds}s at {sr} Hz…")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    sf.write(out_wav, audio, sr)
    print(f"✅ Saved: {out_wav}")
    return out_wav

def vosk_transcribe(wav_path, sr=SAMPLE_RATE, print_words=False):
  """Stream the WAV into Vosk and return the final transcript."""
  _ensure_model()
  rec = KaldiRecognizer(MODEL, sr)
  rec.SetWords(True)

  # Read in chunks to mimic streaming
  with wave.open(wav_path, "rb") as wf:
    assert wf.getframerate() == sr, f"Expected {sr}Hz, got {wf.getframerate()}Hz"
    assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()}ch"
    while True:
      data = wf.readframes(4000)
      if len(data) == 0:
        break
      rec.AcceptWaveform(data)

  # Final result
  result = json.loads(rec.Result())
  text = (result.get("text") or "").strip()

  if print_words and "result" in result:
    # word-level timestamps|
    for w in result["result"]:
      print(f"{w['word']:>15s}  {w['start']:6.2f}–{w['end']:6.2f}  conf={w.get('conf',0):.2f}")

  return text

def record_and_transcribe(seconds=5, save_prefix="take1"):
    wav = f"{save_prefix}.wav"
    record_wav(seconds=seconds, out_wav=wav, sr=SAMPLE_RATE)
    print("Transcribing…")
    text = vosk_transcribe(wav, sr=SAMPLE_RATE, print_words=False)
    return {"audio_path": wav, "text": text}



# Extract location
def extract_location(answer):
    print("\n" + answer["text"])
    
    doc = nlp_trained(answer["text"])
    
    for ent in doc.ents:
        print(f"{ent.label_}: {ent.text}")

    return doc



# Streamlit
import streamlit as st
import base64
from streamlit.components.v1 import html
import time

# Initialize session state for image swap
# if 'mic' not in st.session_state:
#     st.session_state.mic = True

def autoplay_audio(path: str):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    html(
        f"""
        <audio autoplay>
          <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
        </audio>
        """,
        height=0,
    )

# def toggle_mic():
#     st.session_state.mic = not st.session_state.mic

def start_recording():
    response = record_and_transcribe(seconds=5, save_prefix="./media/user_answer")
    ans_doc = extract_location(response)
    for ent in ans_doc.ents:
        st.markdown(f"{ent.label_}: {ent.text}")

column1, column2, column3 = st.columns([0.6, 1.8, 0.6])
with column2:
    st.title("Lodge Police Report")

st.title("")
st.title("")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    # Button lives in col2 (not c2) and is rendered first
    if st.button("Begin", use_container_width=True):
        autoplay_audio("./media/formqn1.mp3")
        time.sleep(2.5)
        # toggle_mic()
        start_recording()
        

    # Now create inner columns for layout of the image
    # c1, c2, c3 = st.columns([1, 1, 1])
    # with c2:
    #     if st.session_state.mic:
    #         st.image("./media/mic-idle.png", caption="idle")
    #     else:
    #         st.image("./media/mic-inuse.png", caption="listening")





