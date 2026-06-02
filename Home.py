import streamlit as st

st.set_page_config(page_title="Song Emotion Analyzer", layout="wide")

st.title("🎵 Emotion Classification from Spotify Lyrics")
st.markdown("---")

st.header("Project Overview")
st.write("""
Proyek Machine Learning ini bertujuan untuk mengklasifikasikan emosi pada lirik lagu menggunakan teknik Natural Language Processing (NLP). 
Model ini dilatih menggunakan dataset dari Spotify dan memprediksi salah satu dari 6 kategori emosi:
**Joy, Sadness, Anger, Fear, Love,** atau **Surprise**.

*Pipeline* pemrosesan menggunakan **TF-IDF Vectorization** yang dikombinasikan dengan **Linear SVM Classifier** untuk mencapai prediksi emosi yang akurat pada data teks.
""")

st.header("Dataset")
st.write("""
**Sumber:** Kaggle - 500K Spotify Dataset  
Dataset ini berisi sekitar 500.000 lagu Spotify dengan metadata terkait termasuk lirik dan label emosi. Dalam proyek ini:
- Dataset difilter dan diseimbangkan.
- Menggunakan pembagian Train/Test (default 80/20).
- Stratified sampling memastikan representasi emosi seimbang di seluruh set.
""")

st.header("Kategori Emosi")
col1, col2 = st.columns(2)
with col1:
    st.info("🟢 **Joy:** Emosi bahagia dan positif")
    st.error("🔴 **Anger:** Frustrasi atau agresif")
    st.warning("😲 **Surprise:** Hal yang tak terduga")
with col2:
    st.info("🔵 **Sadness:** Emosi melankolis atau sedih")
    st.warning("🟣 **Fear:** Cemas atau takut")
    st.success("💖 **Love:** Romantis atau penuh kasih")

st.markdown("---")
st.caption("Silakan navigasi melalui menu di sebelah kiri.")