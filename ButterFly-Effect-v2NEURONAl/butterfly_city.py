import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import numpy as np

# --- 1. CONFIGURATION ET CHARGEMENT DES ARTEFACTS ---

# Le seuil de décision optimal que nous avons choisi
DECISION_THRESHOLD = 0.7

# Noms des fichiers sauvegardés par le script d'entraînement
MODEL_PATH = 'butterfly_expert_model.keras'
SCALER_PATH = 'expert_scaler.joblib'
COLUMNS_PATH = 'expert_model_columns.json'
FEATURES_CONFIG_PATH = 'feature_config.json'

print("Chargement des artefacts du modèle...")

try:
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = json.load(open(COLUMNS_PATH))
    feature_config = json.load(open(FEATURES_CONFIG_PATH))
    print("-> Modèle, scaler et configurations chargés avec succès.")
except FileNotFoundError as e:
    print(f"Erreur : Fichier manquant -> {e.filename}")
    print("Assurez-vous que les fichiers du modèle sont dans le même dossier que ce script.")
    exit()

# --- 2. FONCTION DE PRÉDICTION ---

def predict_incident(new_data: dict):
    """
    Prépare les données d'un nouvel incident, exécute la prédiction et retourne un verdict.
    
    :param new_data: Un dictionnaire contenant les caractéristiques brutes du nouvel incident.
    :return: Un dictionnaire avec la décision et le score de confiance.
    """
    print("\nPréparation des données du nouvel incident pour la prédiction...")
    
    # Créer un DataFrame à partir des données d'entrée
    df = pd.DataFrame([new_data])
    
    # S'assurer que les colonnes numériques sont bien de type float avant le scaling
    numerical_cols = feature_config['numerical_features']
    df[numerical_cols] = df[numerical_cols].astype(float)

    # Recréer toutes les colonnes catégorielles attendues par le modèle, initialisées à 0
    all_categorical_features = feature_config.get('categorical_features_one_hot', [])
    for col in all_categorical_features:
        df[col] = 0

    # Activer les bonnes colonnes One-Hot basées sur les données d'entrée
    for key, value in new_data.items():
        one_hot_col_name = f"{key}_{value}"
        if one_hot_col_name in df.columns:
            df.loc[0, one_hot_col_name] = 1

    # S'assurer que les colonnes numériques sont bien présentes
    for col in numerical_cols:
         if col not in df.columns:
             df[col] = 0

    # Ré-ordonner les colonnes pour correspondre PARFAITEMENT à l'entrée du modèle
    try:
        df_processed = df[model_columns]
    except KeyError as e:
        print(f"Erreur de colonne : La colonne {e} est attendue par le modèle mais n'a pas été trouvée.")
        return None

    # Appliquer le scaler sur les colonnes numériques
    df_processed.loc[:, numerical_cols] = scaler.transform(df_processed[numerical_cols])
    
    print("-> Données prêtes. Lancement de la prédiction...")
    
    # Faire la prédiction
    prediction_proba = model.predict(df_processed, verbose=0)[0][0]
    
    # Interpréter le résultat avec notre seuil optimisé
    decision = "Utile" if prediction_proba > DECISION_THRESHOLD else "Inutile"
    confidence = prediction_proba if decision == "Utile" else 1 - prediction_proba
    
    return {
        "decision": decision,
        "raw_probability": float(prediction_proba),
        "confidence_score": f"{confidence * 100:.2f}%",
        "threshold_used": DECISION_THRESHOLD
    }

# --- 3. EXEMPLE D'UTILISATION ---

if __name__ == "__main__":
    
    # Scénario : Un vol vient d'être signalé (0 jours) à 22h un vendredi (weekend)
    # par un touriste dans une zone touristique. L'utilisateur est relativement proche.
    
    new_incident_data = {
        # Caractéristiques numériques
        "distance_km": 1.2,
        "days_since_incident": 0,
        "hour_of_day": 22,
        "user_engagement_rate": 0.7,
        "user_num_friends": 45,
        "user_connections": 180,
        "urgency_score": (1 / (1.2 + 0.1)) * (1 / (0 + 1)), # Doit être calculé
        
        # Caractéristiques catégorielles (valeurs brutes)
        "user_type": "Tourist",
        "incident_type": "Vol",
        "day_of_week": 4, # Vendredi
        "is_weekend": 1,
        "incident_zone_type": "touristic"
    }

    print("\n--- NOUVEL INCIDENT À ANALYSER ---")
    for key, value in new_incident_data.items():
        print(f"  - {key}: {value}")
    
    # Obtenir la prédiction
    prediction_result = predict_incident(new_incident_data)
    if prediction_result:
        print("\n--- RÉSULTAT DE L'ANALYSE ---")
        print(f"Décision de l'IA : L'alerte est jugée '{prediction_result['decision']}'.")
        print(f"Probabilité brute : {prediction_result['raw_probability']:.4f} (Seuil > {prediction_result['threshold_used']})")
        print(f"Confiance du modèle dans sa décision : {prediction_result['confidence_score']}")
        
        
