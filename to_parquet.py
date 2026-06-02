import pandas as pd

print("1. Membaca dataset asli (csv)...")
# Pastikan path ini mengarah ke file CSV asli yang 1 GB
df = pd.read_csv('dataset/spotify_dataset.csv')

print(f"2. Mengambil sample 50.000 baris...")
df_sample = df.sample(n=100000, random_state=42)

print("3. Menyimpan ke format Parquet...")
# Menyimpan data langsung ke folder dataset dengan ekstensi .parquet
df_sample.to_parquet('dataset/spotify_dataset_mini.parquet', index=False)

print("🎉 Selesai! Coba cek ukuran file .parquet barunya di folder dataset.")