import streamlit as st
import random
import re
from groq import Groq

# Set Konfigurasi Halaman
st.set_page_config(
    page_title="Humanize AI & AI Detector",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Text Detector & Humanizer")
st.write("Deteksi teks buatan AI dan ubah gaya bahasanya menjadi seperti tulisan manusia alami.")

# Sidebar untuk API Key Gratis (Groq)
st.sidebar.header("⚙️ Pengaturan API")
groq_api_key = st.sidebar.text_input(
    "Masukkan Groq API Key (Gratis):", 
    type="password",
    help="Dapatkan API Key gratis di https://console.groq.com/keys"
)

st.sidebar.markdown("""
---
💡 **Petunjuk Bikin API Key Gratis:**
1. Buka [console.groq.com](https://console.groq.com)
2. Login / Daftar dengan akun Google
3. Klik **API Keys** -> **Create API Key**
4. Tempelkan kunci tersebut di kolom atas.
""")

# Fungsi Analisis AI Detector (Pola & Kosa Kata Khas AI)
def analyze_ai_text(text):
    words = text.split()
    total_words = len(words)
    
    if total_words < 10:
        return 0, "Teks terlalu pendek untuk dianalisis."
    
    ai_indicators = [
        "furthermore", "moreover", "in conclusion", "furthermore", "testament",
        "delve", "tapestry", "beacon", "vital role", "furthermore", "oleh karena itu",
        "selain itu", "sebagai kesimpulan", "menjelajahi", "penting untuk dicatat"
    ]
    
    score = 15
    matches = [w for w in ai_indicators if w in text.lower()]
    score += len(matches) * 12
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        lens = [len(s.split()) for s in sentences]
        avg_len = sum(lens) / len(lens)
        if all(abs(l - avg_len) < 3 for l in lens):
            score += 25

    final_score = min(score, 98)
    return final_score, matches

# Tabs Fitur Utama
tab1, tab2 = st.tabs(["🔍 AI Content Detector", "✍️ Humanize AI Text"])

# TAB 1: AI DETECTOR
with tab1:
    st.subheader("Deteksi Apakah Teks Dibuat oleh AI")
    input_text_det = st.text_area("Masukkan teks untuk dianalisis:", height=200, key="det_input")
    
    if st.button("Analisis Teks", type="primary"):
        if not input_text_det.strip():
            st.warning("Harap masukkan teks terlebih dahulu!")
        else:
            ai_percentage, matches = analyze_ai_text(input_text_det)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Skor AI Probabilitas", f"{ai_percentage}%")
            
            with col2:
                if ai_percentage > 60:
                    st.error("🚨 **Kemungkinan Besar Buatan AI**: Teks ini memiliki struktur dan kosakata khas AI.")
                elif ai_percentage > 30:
                    st.warning("⚠️ **Campuran (Hybrid)**: Teks mengandung beberapa pola tulisan AI.")
                else:
                    st.success("✅ **Tulisan Manusia**: Teks terlihat alami dan ditulis oleh manusia.")
            
            if matches:
                st.caption(f"Kata/Frasa khas AI yang terdeteksi: `{', '.join(set(matches))}`")

# TAB 2: HUMANIZE TEXT
with tab2:
    st.subheader("Ubah Teks AI Menjadi Tulisan Manusia Alami")
    input_text_hum = st.text_area("Masukkan teks AI yang ingin di-humanize:", height=200, key="hum_input")
    
    tone = st.selectbox("Pilih Gaya Bahasa:", ["Santai & Natural", "Akademik Natural", "Profesional / Bisnis"])
    
    if st.button("Humanize Text ✨", type="primary"):
        if not groq_api_key:
            st.error("🔑 Harap masukkan Groq API Key gratis Anda di sidebar sebelah kiri!")
        elif not input_text_hum.strip():
            st.warning("Harap masukkan teks terlebih dahulu!")
        else:
            try:
                client = Groq(api_key=groq_api_key)
                
                prompt = f"""
                Tugas Anda adalah merubah teks berikut agar lolos AI Detector (seperti Turnitin / ZeroGPT) dan terdengar 100% seperti buatan manusia alami.
                
                Instruksi:
                1. Ubah frasa/kosa kata kaku khas AI (seperti "selain itu", "penting untuk dicatat", "tapestry", "delve") menjadi kata sehari-hari.
                2. Variasikan panjang kalimat (gabungkan beberapa kalimat pendek, buat beberapa kalimat yang lebih panjang).
                3. Gunakan gaya bahasa: {tone}.
                4. Jaga agar arti dan fakta utama tetap sama persis.
                5. HANYA berikan hasil teks yang sudah di-humanize tanpa pengantar/penjelasan tambahan.
                
                Teks asal:
                {input_text_hum}
                """
                
                with st.spinner("Sedang memproses teks..."):
                    completion = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                    )
                    
                    result = completion.choices[0].message.content
                    
                    st.subheader("Hasil Humanize:")
                    st.success("Selesai!")
                    st.text_area("Teks Hasil:", value=result, height=220)
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
