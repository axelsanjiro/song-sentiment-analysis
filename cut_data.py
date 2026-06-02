import pandas as pd

print("Sedang membaca dataset asli (proses ini butuh waktu beberapa detik)...")
# Pastikan nama file CSV aslinya sesuai
df = pd.read_csv('dataset/spotify_dataset.csv')

print(f"Jumlah baris dataset asli: {len(df)}")

# Mengambil sampel data secara acak (Random Sampling)
# Mengambil 50.000 baris sangat direkomendasikan agar aman masuk GitHub
df_mini = df.sample(n=50000, random_state=42)

print(f"Data berhasil dipotong! Jumlah baris sekarang: {len(df_mini)}")

print("Menyimpan ke file CSV baru...")
# Menyimpan dengan nama baru agar file asli tidak tertimpa
df_mini.to_csv('spotify_dataset_mini.csv', index=False)

print("Selesai! File 'spotify_dataset_mini.csv' siap digunakan.")