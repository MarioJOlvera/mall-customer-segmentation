import pandas as pd
import joblib 

from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 

DATA_PATH = "data/raw/Mall_Customers.csv"
MODEL_PATH = "models/cluster_pipeline.joblib" 

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"] 

def train(): 

	df = pd.read_csv(DATA_PATH)
	X = df[FEATURES]

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X) 

	pca = PCA(n_components = 2) 
	X_pca = pca.fit_transform(X_scaled) 

	kmeans = KMeans(n_clusters = 5, random_state = 42, n_init = 10) 
	kmeans.fit(X_scaled)

	artifact = {
		"scaler": scaler, 
		"pca": pca, 
		"kmeans": kmeans, 
		"features": FEATURES, 
	}

	joblib.dump(artifact, MODEL_PATH)
	print(f"Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__": 
	train() 


