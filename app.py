import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. SETUP MODEL AI (GRATIS DARI HUGGINGFACE)
# ==========================================
@st.cache_resource
def load_models():
    # Menggunakan model T5 untuk menulis ulang/parafrase teks
    humanizer = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
    # Menggunakan model pendeteksi apakah teks buatan AI atau Manusia
    ai_detector = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")
    return humanizer, ai_detector

humanizer_pipeline, detector_pipeline = load_models()

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
    skor_tertinggi = max(skor_kemiripan[0]) * 100
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
                label = hasil_deteksi[0]['label']
                skor_ai = hasil_deteksi[0]['score'] * 100
                
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
                prompt_input = f"paraphrase: {teks_pengguna}"
                hasil_humanis = humanizer_pipeline(prompt_input, max_length=len(teks_pengguna)+50)[0]['generated_text']
                
                st.subheader("Hasil Humanisasi:")
                st.write(hasil_humanis)
                st.info("💡 Teks di atas telah ditulis ulang oleh AI agar struktur kalimatnya tidak kaku.")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu!")
              
