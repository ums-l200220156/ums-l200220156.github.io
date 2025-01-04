import pandas as pd

# File input dan output
input_file = "Tugas.csv"  # Ganti dengan nama file Anda
output_file = "data_group_cleaned.csv"

# Membaca file CSV
data = pd.read_csv(input_file)

# Menambahkan nama kolom (jika tidak ada header)
data.columns = ['Date', 'Time', 'Sender', 'Message']

# Membersihkan data
# 1. Hapus baris di mana kolom 'Sender' mengandung placeholder atau simbol
data = data[~data['Sender'].str.contains(r'^[-:a-z]$', na=False)]

# 2. Hapus pesan yang mengandung "<Media tidak disertakan>"
data = data[~data['Message'].str.contains(r'<Media tidak disertakan>', na=False)]

# Simpan data yang telah dibersihkan ke file CSV baru
data.to_csv(output_file, index=False)

print(f"Data bersih telah disimpan ke file: {output_file}")
