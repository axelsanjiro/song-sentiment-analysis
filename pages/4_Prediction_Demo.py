import streamlit as st
import joblib
import re

st.set_page_config(page_title="Prediction Demo", layout="centered")

# --- FUNGSI PREPROCESSING TEKS ---
# Harus sama persis dengan yang ada di proses training
def remove_tags(text):
    return re.sub(r'\[.*?\]', '', str(text))

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def reduce_repetition(text):
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

def preprocess_lyrics(text):
    text = remove_tags(text)
    text = basic_clean(text)
    text = reduce_repetition(text)
    return text

# --- LOAD MODEL ---
def load_model():
    # 1. Cek apakah model ada di session state (baru saja di-train)
    if 'trained_model' in st.session_state:
        return st.session_state['trained_model']
    
    # 2. Kalau tidak ada, coba load dari file .pkl lokal
    try:
        return joblib.load('emotion_model.pkl')
    except Exception:
        return None

pipeline = load_model()

# --- UI ELEMENTS ---
emotion_ui = {
    'joy': {'icon': '🟢', 'label': 'Joy (Bahagia)'},
    'sadness': {'icon': '🔵', 'label': 'Sadness (Sedih)'},
    'anger': {'icon': '🔴', 'label': 'Anger (Marah)'},
    'fear': {'icon': '🟣', 'label': 'Fear (Takut)'},
    'love': {'icon': '💖', 'label': 'Love (Cinta)'},
    'surprise': {'icon': '😲', 'label': 'Surprise (Terkejut)'}
}

st.title("Prediction Demo")
st.write("Uji coba model AI dengan lirik lagumu sendiri. Apakah lagunya bernuansa gembira, sedih, atau penuh amarah?")
st.markdown("---")

# --- INPUT AREA ---
lyrics_input = st.text_area(
    "Masukkan Lirik Lagu di Sini:", 
    height=250, 
    placeholder="[Verse 1]\nI'm so happy today...\n\n(Hapus tag seperti [Verse] atau [Chorus] opsional, sistem akan membersihkannya otomatis)"
)

# --- INFERENCE SECTION ---
if st.button("Analisis Emosi", type="primary", width=True):
    if not pipeline:
        st.error("Model belum siap! Silakan buka halaman **Model Training** terlebih dahulu dan jalankan proses training untuk membuat file 'emotion_model.pkl'.")
    elif lyrics_input.strip() == "":
        st.warning("lirik lagunya belum diisi!")
    else:
        with st.spinner("Sedang menganalisis emosi dari lirik..."):
            # 1. Preprocessing Input
            cleaned_lyrics = preprocess_lyrics(lyrics_input)
            
            # 2. Validasi panjang kata
            if len(cleaned_lyrics.split()) <= 5:
                st.error("Lirik terlalu pendek! Masukkan lirik yang lebih panjang (lebih dari 5 kata yang valid) agar prediksi model akurat.")
            else:
                # 3. Prediksi dengan Model
                prediction = pipeline.predict([cleaned_lyrics])[0]
                
                # 4. Tampilkan Hasil
                st.success("Analisis berhasil diselesaikan!")
                
                # Mengambil icon dan label dari dictionary
                ui_data = emotion_ui.get(prediction, {'label': prediction.capitalize()})
                
                # Menampilkan hasil dengan layout kolom agar posisinya di tengah
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: #f9f9f9; color: #1e1e1e;">
                            <h3 style="text-align: center; margin-bottom: 5px; color: #1e1e1e;">Emosi Dominan</h3>
                            <h1 style="text-align: center; margin-top: 0; color: #1e1e1e;">{ui_data['icon']} {ui_data['label']}</h1>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Opsi untuk melihat hasil teks yang sudah dibersihkan
                with st.expander("Lihat Teks yang Diproses (Hasil Cleaning)"):
                    st.write(cleaned_lyrics)
                    st.caption(f"Jumlah kata valid: {len(cleaned_lyrics.split())} kata.")

st.markdown("---")
st.caption("Final Project NLP | Streamlit UI")