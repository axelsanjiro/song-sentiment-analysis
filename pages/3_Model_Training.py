import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

st.set_page_config(page_title="Model Training", layout="wide")
st.title("Model Training")
st.write("Latih model Linear Support Vector Classification (LinearSVC) menggunakan data yang sudah diproses.")

# --- CEK KETERSEDIAAN DATA ---
if 'cleaned_data' not in st.session_state:
    st.warning("Data belum diproses! Silakan kembali ke menu **Preprocessing** terlebih dahulu dan klik tombol 'Proses Dataset & Simpan ke Session'.")
else:
    df = st.session_state['cleaned_data']
    st.success(f"Ditemukan dataset yang siap dilatih sebanyak {len(df)} baris.")
    
    st.markdown("---")
    
    # --- PENGATURAN HYPERPARAMETER ---
    st.markdown("### Pengaturan Hyperparameter")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Pembagian Data (Train/Test Split)**")
        test_size = st.slider("Ukuran Data Uji (Test Size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05, 
                            help="Berapa persen data yang akan disisihkan untuk menguji model? (0.2 = 20%)")
        
    with col2:
        st.write("**Konfigurasi Model LinearSVC**")
        c_value = st.slider("Regularization Parameter (C)", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                            help="Semakin kecil nilai C, semakin kuat regularisasinya (mencegah overfitting).")

    st.markdown("---")
    
    # --- PROSES TRAINING ---
    if st.button("Mulai Latih Model", type="primary"):
        with st.spinner("Sedang membagi data dan melatih model... Proses ini mungkin butuh beberapa detik."):
            
            # 1. Train-Test Split (Memastikan stratify agar distribusi emosi seimbang)
            X_train, X_test, y_train, y_test = train_test_split(
                df['text'], 
                df['emotion'], 
                test_size=test_size, 
                stratify=df['emotion'], 
                random_state=42
            )
            
            # 2. Setup Pipeline (Sesuai dengan main.ipynb)
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True, 
                    stop_words='english', 
                    ngram_range=(1,2),
                    min_df=3, 
                    max_df=0.95, 
                    sublinear_tf=True,
                    max_features=100000
                )),
                ("model", LinearSVC(class_weight='balanced', C=c_value, dual='auto'))
            ])
            
            # 3. Fitting Model
            pipeline.fit(X_train, y_train)
            
            # 4. Evaluasi Model
            y_pred = pipeline.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            # 5. Simpan Model menggunakan Joblib
            joblib.dump(pipeline, "emotion_model.pkl")
            
            st.session_state['trained_model'] = pipeline 
            
            st.success("🎉 Training Selesai! Model berhasil disimpan...")
            
            # --- TAMPILAN HASIL ---
            st.markdown("### Hasil Evaluasi Model (Classification Report)")
            
            # Menampilkan akurasi utama
            accuracy = report_df.loc['accuracy', 'precision'] # Di sklearn output dict, accuracy ada di baris tersendiri
            st.metric(label="Akurasi Keseluruhan", value=f"{accuracy * 100:.2f}%")
            
            # Menampilkan tabel metrik detail
            st.write("Detail performa per kategori emosi:")
            # Hapus baris 'accuracy' dari dataframe untuk tabel agar lebih rapi
            report_table = report_df.drop('accuracy', errors='ignore')
            
            # Format tabel agar angkanya lebih enak dibaca
            st.dataframe(
                report_table.style.format("{:.3f}")
                .background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
            )
            
            st.info("Model sudah tersimpan! Sekarang kamu bisa pindah ke halaman **Prediction Demo** untuk mencoba memprediksi lirik lagumu sendiri.")