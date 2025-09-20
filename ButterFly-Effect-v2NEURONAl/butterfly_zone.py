# app.py
# Ton application principale

import butterfly # On importe notre module de prédiction

def on_new_incident_reported(incident_data):
    """
    Cette fonction est appelée quand un nouvel incident est signalé dans ton app.
    """
    print("Nouvel incident reçu, analyse en cours...")
    
    # 1. On appelle la fonction de notre module
    prediction = butterfly.predict_incident(incident_data)
    
    # 2. On utilise le résultat
    if prediction:
        print(f"Décision de l'IA : {prediction['decision']}")
        print(f"Score brut : {prediction['raw_probability']}")

        # Ici, tu peux déclencher la logique d'alerte dynamique
        if prediction['raw_probability'] > 0.85:
            # Créer une zone d'alerte de 2km, trouver les utilisateurs, etc.
            pass

# --- Simulation ---
# Un nouvel incident arrive sous forme de dictionnaire
new_incident = {
    "distance_km": 0.8,
    "days_since_incident": 0,
    "hour_of_day": 2, # 2h du matin
    "user_engagement_rate": 0.9,
    "user_num_friends": 200,
    "user_connections": 800,
    "urgency_score": (1 / (0.8 + 0.1)) * (1 / (0 + 1)),
    "user_type": "Local",
    "incident_type": "Agression",
    "day_of_week": 5, # Samedi
    "is_weekend": 1,
    "incident_zone_type": "nightlife"
}

on_new_incident_reported(new_incident)
