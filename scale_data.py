import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def load_data_group(file_path, num_docs=None):
    """
    Memuat data dari file CSV.
    Args:
        file_path (str): Lokasi file CSV.
        num_docs (int): Jumlah dokumen yang akan dimuat. Jika None, muat semua dokumen.
    Returns:
        list: List dokumen teks.
    """
    # Membaca file CSV
    data = pd.read_csv(file_path)
    
    # Ambil kolom 'Message' dan hilangkan NaN
    data['Message'] = data['Message'].fillna('')  # Ganti NaN dengan string kosong
    
    # Ambil dokumen sesuai jumlah yang diminta
    if num_docs:
        return data['Message'][:num_docs].tolist()
    return data['Message'].tolist()

def make_matrix(docs, binary=False, min_df=10, max_df=0.1):
    """
    Membuat matriks fitur berdasarkan dokumen.
    Args:
        docs (list): Daftar dokumen teks.
        binary (bool): Apakah menggunakan binerisasi.
        min_df (int): Minimum dokumen yang mengandung fitur.
        max_df (float): Maksimum dokumen yang mengandung fitur (sebagai proporsi).
    Returns:
        tuple: Matriks fitur dan daftar fitur.
    """
    # Membuat CountVectorizer
    vec = CountVectorizer(min_df=min_df, max_df=max_df, binary=binary)
    
    # Transformasi dokumen menjadi matriks
    mtx = vec.fit_transform(docs)
    feature_names = vec.get_feature_names_out()
    
    return mtx, feature_names

if __name__ == "__main__":
    # Path ke file data
    file_path = '/mnt/d/ums-l200220156.github.io/data_group_cleaned.csv'
    
    # Uji fungsi load_data_group
    try:
        print("Memuat data...")
        docs = load_data_group(file_path, num_docs=100)  # Ambil 100 dokumen pertama
        print(f"Jumlah dokumen yang dimuat: {len(docs)}")
    except Exception as e:
        print(f"Error saat memuat data: {e}")
    
    # Uji fungsi make_matrix
    try:
        print("Membuat matriks fitur...")
        mtx, feature_names = make_matrix(docs)
        print(f"Dimensi matriks fitur: {mtx.shape}")
        print(f"Fitur pertama: {feature_names[:10]}")
    except Exception as e:
        print(f"Error saat membuat matriks fitur: {e}")
