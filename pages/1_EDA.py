import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA", layout="wide")
st.title("Exploratory Data Analysis (EDA)")

# Fungsi cache agar dataset tidak diload ulang setiap perpindahan halaman
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('dataset/spotify_dataset_mini.parquet') 
        
        # Normalisasi label
        df['emotion'] = df['emotion'].astype(str).str.lower().str.strip()
        mapping = {'angry': 'anger', 'confusion': 'surprise', 'interest': 'surprise'}
        df['emotion'] = df['emotion'].replace(mapping)
        valid_labels = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
        df = df[df['emotion'].isin(valid_labels)]
        
        return df
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return None

df = load_data()

if df is not None:
    st.write("### Preview Dataset Awal")
    st.dataframe(df[['Artist(s)', 'song', 'emotion', 'text']].head(10))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Distribusi Kategori Emosi")
        emotion_counts = df['emotion'].value_counts().reset_index()
        emotion_counts.columns = ['Emotion', 'Count']
        fig_bar = px.bar(emotion_counts, x='Emotion', y='Count', color='Emotion', text_auto=True)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        st.write("### Persentase Emosi (Pie Chart)")
        fig_pie = px.pie(emotion_counts, names='Emotion', values='Count', hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.write("### Analisis Panjang Teks (Jumlah Kata)")
    # Hitung jumlah kata per baris untuk visualisasi
    sample_df = df.sample(10000, random_state=42) # Ambil sample agar render cepat
    sample_df['word_count'] = sample_df['text'].apply(lambda x: len(str(x).split()))
    fig_hist = px.histogram(sample_df, x='word_count', color='emotion', nbins=50, 
                            title="Distribusi Panjang Lirik per Emosi (Sample 10k data)")
    st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.warning("Pastikan file 'spotify_dataset_mini.csv' berada di direktori utama.")