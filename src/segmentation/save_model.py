# src/segmentation/save_model.py

import joblib

def save_artifact(artifact, path):
    joblib.dump(artifact, path)
    print(f"Artefacto guardado en {path}")
