import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib

import matplotlib.pyplot as plt

import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import whisper

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

# 📄 Extract data from PDF files
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# ✂️ Split the data into smaller text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# 🔗 Download HuggingFace embeddings
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

# 🎤 Record voice input from microphone
def record_audio(filename="mic_input.wav", duration=5, fs=44100):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write_wav(filename, fs, audio)
    print("✅ Recording complete.")
    return filename

# 🧠 Transcribe voice using Whisper
def transcribe_audio(filename):
    print("🧠 Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    text = result["text"]
    print("🗣️ You said:", text)
    return text

# 📊 Plot disease trend graph (mock data)
def fetch_and_plot_disease_data(query):
    disease = extract_disease_from_query(query)

    # 🧹 Auto-delete old graph images (>5 min old)
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    else:
        now = time.time()
        for filename in os.listdir(static_dir):
            file_path = os.path.join(static_dir, filename)
            if filename.endswith(".png") and os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > 300:  # 5 minutes
                    os.remove(file_path)
                    print(f"🗑️ Deleted old file: {file_path}")

    # Generate mock time-series data
    years = np.arange(2013, 2024)
    values = np.random.randint(50, 300, size=len(years))

    # Create line plot
    plt.figure(figsize=(10, 5))
    plt.plot(years, values, marker='o', linestyle='-', color='skyblue')
    plt.title(f"{disease.title()} Cases Over the Last 10 Years")
    plt.xlabel("Year")
    plt.ylabel("Cases")
    plt.grid(True)

    # Save new plot
    filename = f"{static_dir}/{uuid.uuid4().hex}.png"
    plt.savefig(filename)
    plt.close()

    return filename


# 🧠 Extract disease keyword from user query
def extract_disease_from_query(query):
    keywords = [ "fever", "cold", "cough", "flu", "covid", "asthma", "allergy", "diabetes", "hypertension", "cancer", "malaria", "dengue", "chikungunya", "typhoid", "hepatitis", "tuberculosis", "arthritis", "anemia", "measles", "mumps", "chickenpox", "epilepsy", "pneumonia", "stroke", "heart attack", "depression", "anxiety", "obesity", "kidney disease", "liver disease", "HIV", "AIDS", "thyroid", "eczema", "psoriasis", "ulcer", "migraine", "polio", "autism", "parkinson's", "alzheimer's", "bronchitis", "glaucoma", "conjunctivitis", "UTI", "meningitis", "hepatitis A", "hepatitis B", "hepatitis C", "jaundice"]
    for word in keywords:
        if word.lower() in query.lower():
            return word
    return "disease"
