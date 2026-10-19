import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. SETUP MODEL AI (GRATIS DARI HUGGINGFACE)
# ==========================================
@st.cache_resource
def load_models():
    # Menggunakan AutoModel langsung untuk menghindari bug pipeline di Python 3.14
    humanizer_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    humanizer_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    
    # Menggunakan model pendeteksi AI yang sudah diperbaiki alamatnya
    ai_detector = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")
    
    return humanizer_tokenizer, humanizer_model, ai_detector

h_tokenizer, h_model, detector_pipeline = load_models()

# ==========================================
# 2. CONTOH DATABASE TEKS (UNTUK CEK PLAGIASI)
# ==========================================
DATABASE_TEKS = [
    "Kecerdasan buatan atau AI adalah teknologi yang mensimulasikan kecerdasan manusia.",
    "Pembelajaran mesin merupakan cabang dari kecerdasan buatan yang fokus pada data.",
    "Pendidikan di era digital membutuhkan adaptasi teknologi yang cepat dan merata."
]

def cek_plagiasi(teks_input):
    if not teks_input.strip():
        return 0.0
    semua_teks = [teks_input] + DATABASE_TEKS
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(semua_teks)
    skor_kemiripan = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    skor_tertinggi = max(skor_kemiripan) * 100
    return skor_tertinggi

# ==========================================
# 3. TAMPILAN APLIKASI WEB (STREAMLIT)
# ==========================================
st.set_page_config(page_title="HSW PurePen AI", layout="centered")

st.title("✨ HSW PurePen AI Gratis")
st.write("Aplikasi buatan Herni Satria Wulandari untuk cek plagiasi tingkat tinggi dan memanusiakan teks AI.")

# Kolom Input Teks
teks_pengguna = st.text_area("Masukkan Teks Anda di Sini:", height=200, placeholder="Ketik atau tempel teks buatan AI di sini...")

col1, col2 = st.columns(2)

# TOMBOL 1: CEK PLAGIASI & DETEKSI AI
with col1:
    if st.button("📊 Analisis Teks", use_container_width=True):
        if teks_pengguna:
            with st.spinner("Menganalisis teks..."):
                skor_plagiasi = cek_plagiasi(teks_pengguna)
                hasil_deteksi = detector_pipeline(teks_pengguna)
                label = hasil_deteksi['label']
                skor_ai = hasil_deteksi['score'] * 100
                
                st.subheader("Hasil Analisis:")
                st.metric(label="Skor Plagiasi (Kemiripan)", value=f"{skor_plagiasi:.2f}%")
                
                if label == "Fake" or "AI" in label:
                    st.error(f"Terdeteksi Buatan AI: {skor_ai:.2f}%")
                else:
                    st.success(f"Terdeteksi Buatan Manusia: {skor_ai:.2f}%")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu!")

# TOMBOL 2: HUMANISASI TEKS
with col2:
    if st.button("🤖 Humanize Teks", use_container_width=True):
        if teks_pengguna:
            with st.spinner("Mengubah teks menjadi lebih manusiawi..."):
                # Proses pembuatan teks manusiawi menggunakan tokenization manual yang lebih stabil
                text_input = "paraphrase: " + teks_pengguna
                inputs = h_tokenizer(text_input, return_tensors="pt", padding=True)
                outputs = h_model.generate(**inputs, max_length=len(teks_pengguna)+50)
                hasil_humanis = h_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                st.subheader("Hasil Humanisasi:")
                st.write(hasil_humanis)
                st.info("💡 Teks di atas telah ditulis ulang oleh AI agar struktur kalimatnya tidak kaku.")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu!")
