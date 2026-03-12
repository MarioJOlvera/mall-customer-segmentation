import joblib 
import pandas as pd

MODEL_PATH = "models/cluster_pipeline.joblib"

def predict_cluster(age, income, spending_score): 

	artifact = joblib.load(MODEL_PATH) 
	
	scaler = artifact["scaler"]
	kmeans = artifact["kmeans"]
	features = artifact["features"]

	X_new = pd.DataFrame(
		[[age, income, spending_score]], 
		columns = features
	) 

	X_new_scaled = scaler.transform(X_new) 
	cluster = kmeans.predict(X_new_scaled)[0]
	return cluster 

if __name__ == "__main__":
	cluster = predict_cluster(age = 29, income = 62, spending_score = 80) 
	print(f"Cluster predict: {cluster}") 


