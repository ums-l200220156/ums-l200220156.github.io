class KuliahInformatika:
    def __init__(self, nama_mahasiswa):
        self.nama_mahasiswa = nama_mahasiswa
        self.spp_dibayar = False
        self.nilai_akhir = None

    def bayar_spp(self):
        self.spp_dibayar = True
        print(f"{self.nama_mahasiswa} telah membayar SPP.")

    def mengikuti_kuliah(self):
        if not self.spp_dibayar:
            print(f"{self.nama_mahasiswa} harus membayar SPP terlebih dahulu.")
            return
        print(f"{self.nama_mahasiswa} mengikuti kuliah Informatika.")

    def dapatkan_nilai(self, nilai):
        self.nilai_akhir = nilai
        print(f"{self.nama_mahasiswa} mendapatkan nilai akhir: {self.nilai_akhir}")

def main():
    mahasiswa = KuliahInformatika("Alfian Candra Irwana")
    
    # Proses mengikuti kuliah
    mahasiswa.bayar_spp()
    mahasiswa.mengikuti_kuliah()
    mahasiswa.dapatkan_nilai("A")

if __name__ == "__main__":
    main()