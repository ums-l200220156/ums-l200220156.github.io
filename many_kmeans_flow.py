from metaflow import FlowSpec, step
from sklearn.cluster import KMeans
from scale_data import load_data_group, make_matrix

class ManyKMeansFlow(FlowSpec):
    
    @step
    def start(self):
        """Memuat data dan membuat matriks fitur."""
        # Path file CSV
        data_path = '/mnt/d/ums-l200220156.github.io/data_group_cleaned.csv'
        
        # Memuat data dari file
        docs = load_data_group(data_path)
        
        # Membuat matriks fitur
        self.mtx, self.feature_names = make_matrix(docs)
        
        print(f"Data dimuat, matriks fitur: {self.mtx.shape}")
        self.next(self.cluster_data)
    
    @step
    def cluster_data(self):
        """Melakukan clustering dengan 3, 4, dan 5 cluster."""
        self.results = {}
        for n_clusters in [3, 4, 5]:
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model.fit(self.mtx)
            self.results[n_clusters] = model.labels_
            print(f"Clustering selesai untuk {n_clusters} cluster.")
        self.next(self.analyze_results)
    
    @step
    def analyze_results(self):
        """Menganalisis hasil clustering."""
        self.cluster_terms = {}
        for n_clusters, labels in self.results.items():
            top_terms = []
            for cluster_id in range(n_clusters):
                # Ambil dokumen dalam cluster
                cluster_docs = self.mtx[labels == cluster_id]
                term_freq = cluster_docs.sum(axis=0).A1
                top_term_indices = term_freq.argsort()[-3:][::-1]
                top_terms.append([self.feature_names[i] for i in top_term_indices])
            self.cluster_terms[n_clusters] = top_terms
            print(f"Top terms untuk {n_clusters} cluster: {top_terms}")
        self.next(self.end)
    
    @step
    def end(self):
        """Mengakhiri flow."""
        print("Clustering selesai.")
        for n_clusters, terms in self.cluster_terms.items():
            print(f"Cluster {n_clusters}:")
            for idx, cluster in enumerate(terms):
                print(f"  Cluster {idx}: {cluster}")

if __name__ == "__main__":
    ManyKMeansFlow()
