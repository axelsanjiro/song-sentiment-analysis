import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Preprocessing", layout="wide")
st.title("Data Preprocessing")
st.write("Atur parameter pembersihan teks sebelum model dilatih.")

# PREPROCESSING 
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

# PREPROCESSING (MAIN PAGE)
st.markdown("### Pengaturan Data")
col1, col2 = st.columns(2)

with col1:
    min_words = st.slider("Minimal Jumlah Kata per Baris", min_value=1, max_value=20, value=5)
    
with col2:
    sample_size = st.number_input("Jumlah Data Sample (Untuk performa web)", min_value=1000, max_value=50000, value=50000, step=10000)

st.markdown("---")

# UJI COBA KUSTOM 
st.write("### Uji Coba Preprocessing pada Teks Kustom")
test_text = st.text_area("Masukkan teks kotor:", "[Verse 1]\nI'm so happy happy happy today!!! 😊 123")
if test_text:
    st.code(f"Hasil: {reduce_repetition(basic_clean(remove_tags(test_text)))}")

st.markdown("---")

# EKSEKUSI DATASET
if st.button("Proses Dataset & Simpan ke Session", type="primary"):
    with st.spinner("Sedang memproses dataset... Ini mungkin memakan waktu sesaat."):
        df = pd.read_csv('dataset/spotify_dataset_mini.csv')
        df['emotion'] = df['emotion'].str.lower().str.strip()
        mapping = {'angry': 'anger', 'confusion': 'surprise', 'interest': 'surprise'}
        df['emotion'] = df['emotion'].replace(mapping)
        valid_labels = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
        df = df[df['emotion'].isin(valid_labels)]
        
        # Ambil sampel dan proses
        df_processed = df[['text', 'emotion']].sample(int(sample_size), random_state=42).copy()
        df_processed['text'] = df_processed['text'].apply(remove_tags)
        df_processed['text'] = df_processed['text'].apply(basic_clean)
        df_processed['text'] = df_processed['text'].apply(reduce_repetition)
        
        # Filter berdasar minimum kata
        df_processed = df_processed[df_processed['text'].str.split().str.len() > min_words]
        
        # Simpan ke session state agar bisa diakses di halaman training
        st.session_state['cleaned_data'] = df_processed
        
        st.success(f"Berhasil! Dataset kini memiliki {len(df_processed)} baris siap latih.")
        st.dataframe(df_processed.head())