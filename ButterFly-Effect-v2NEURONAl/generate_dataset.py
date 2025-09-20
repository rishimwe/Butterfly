# generate_expert_dataset.py

import csv
import random
import math
import json
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from enum import Enum

print("Démarrage de la génération du jeu de données v2 (Expert)...")

# --- MOCK BUTTERFLY (INCHANGÉ) ---
class ButterflyMock:
    def get_feature_vector(self, user, incident, distance):
        # Simule des scores basés sur des règles simples pour plus de réalisme
        social_score = len(set(user.friends) & set(incident.victim_friends)) / len(user.friends) if len(user.friends) > 0 else 0
        gravity_score = 1 - (incident.days_since / 90)
        geo_score = max(0, 1 - (distance / 10)) # Diminue avec la distance, max 10km
        return [geo_score, social_score, random.random(), gravity_score]

butterfly = ButterflyMock()

# --- 1. CONFIGURATION CENTRALISÉE & TYPES ---

class UserType(Enum):
    LOCAL = "Local"
    EXPAT = "Expat"
    TOURIST = "Tourist"

class IncidentType(Enum):
    VOL = "Vol"
    AGRESSION = "Agression"
    OBJET_PERDU = "Objet Perdu"
    DISPARITION = "Disparition"

class Config:
    """Classe de configuration pour tous les paramètres de la simulation."""
    DATASET_SIZE = 50000 # Taille réduite pour une génération plus rapide
    OUTPUT_DIR = Path("./")
    CSV_FILENAME = "lisbon_expert_data.csv"
    CONFIG_FILENAME = "feature_config.json"

    ZONES = {
        "touristic": [{"name": "Baixa/Chiado", "lat": 38.7114, "lon": -9.1407}],
        "residential": [{"name": "Campo de Ourique", "lat": 38.7161, "lon": -9.1678}],
        "business": [{"name": "Parque das Nações", "lat": 38.7684, "lon": -9.0993}],
        "suburbs": [{"name": "Amadora", "lat": 38.7591, "lon": -9.2236}],
        "nightlife": [{"name": "Bairro Alto", "lat": 38.7110, "lon": -9.1453}]
    }
    
    # Probabilités pour la génération
    WEIGHTS = {
        "user_type": [0.6, 0.2, 0.2], # Local, Expat, Tourist
        "incident_type": [0.5, 0.1, 0.35, 0.05], # Vol, Agression, Objet Perdu, Disparition
    }

# --- 2. CLASSES DE MODÈLE (USER, INCIDENT) ---

class User:
    """Représente un utilisateur avec des caractéristiques et comportements."""
    def __init__(self):
        self.user_type = random.choices(
            [UserType.LOCAL, UserType.EXPAT, UserType.TOURIST],
            weights=Config.WEIGHTS["user_type"], k=1
        )[0]
        
        home_zone = "suburbs" if self.user_type == UserType.LOCAL else "residential"
        self.home_location = self._get_random_location_in_zone(home_zone)
        
        self.friends = [random.randint(1000, 10000) for _ in range(random.randint(10, 150))]
        self.connections = random.randint(50, 2000)
        self.engagement_rate = random.uniform(0.1, 0.9)

    def get_current_location(self, hour: int, is_weekend: bool):
        """Détermine un lieu réaliste en fonction du profil et de l'heure."""
        if self.user_type == UserType.TOURIST:
            return self._get_random_location_in_zone("touristic")
        
        if not is_weekend and 8 <= hour <= 18: # En semaine, au travail
            return self._get_random_location_in_zone("business")
        elif hour >= 22 or hour <= 4: # La nuit
            return self._get_random_location_in_zone("nightlife")
        else: # Soir & week-end
            return self._get_random_location_in_zone(random.choice(["residential", "touristic"]))

    def _get_random_location_in_zone(self, zone_type: str):
        zone = random.choice(Config.ZONES[zone_type])
        offset = random.uniform(-0.01, 0.01)
        return {"lat": zone["lat"] + offset, "lon": zone["lon"] + offset}

class Incident:
    """Représente un incident avec un contexte temporel et spatial."""
    def __init__(self):
        self.type = random.choices(
            [it for it in IncidentType], weights=Config.WEIGHTS["incident_type"], k=1
        )[0]
        
        self.date = datetime.now() - timedelta(
            days=random.randint(0, 30), hours=random.randint(0, 23)
        )
        self.hour_of_day = self.date.hour
        self.day_of_week = self.date.weekday() # Lundi=0, Dimanche=6
        self.is_weekend = self.day_of_week >= 5
        self.days_since = (datetime.now() - self.date).days
        
        self.location, self.zone_type = self._get_realistic_location()
        self.victim_friends = [random.randint(1000, 10000) for _ in range(random.randint(10, 150))]

    def _get_realistic_location(self):
        """Détermine un lieu réaliste pour l'incident."""
        if self.type in [IncidentType.AGRESSION, IncidentType.DISPARITION] or (self.hour_of_day >= 22 or self.hour_of_day <= 4):
            zone = "nightlife"
        elif self.type == IncidentType.VOL:
            zone = "touristic"
        else:
            zone = random.choice(["touristic", "residential"])
        
        location = User._get_random_location_in_zone(self, zone)
        return location, zone

# --- 3. ORACLE AMÉLIORÉ ---

class Oracle:
    """Logique experte améliorée pour déterminer la "vérité terrain"."""
    @staticmethod
    def decide_if_useful(incident: Incident, urgency: float, geo: float, social: float, gravity: float):
        if incident.type == IncidentType.DISPARITION:
            return 1 # Toujours utile
        
        if incident.type == IncidentType.AGRESSION:
            if geo > 0.8 or urgency > 0.5 or social > 0.3:
                return 1
        
        if incident.type == IncidentType.VOL:
            if (geo > 0.9 and gravity > 0.5) or (urgency > 0.7 and social > 0.1):
                return 1

        if incident.type == IncidentType.OBJET_PERDU:
            if urgency > 0.9 and geo > 0.95:
                return 1
        
        return 0

# --- 4. GÉNÉRATEUR DE DATASET ---

class DatasetGenerator:
    """Génère le dataset complet et le fichier de configuration des caractéristiques."""
    def __init__(self, config: Config):
        self.config = config
        self.numerical_features = [
            "distance_km", "days_since_incident", "hour_of_day",
            "user_engagement_rate", "user_num_friends", "user_connections",
            "urgency_score"
        ]
        self.categorical_features = {
            "user_type": [ut.value for ut in UserType],
            "incident_type": [it.value for it in IncidentType],
            "day_of_week": list(range(7)),
            "is_weekend": [0, 1],
            "incident_zone_type": list(config.ZONES.keys())
        }

    def _calculate_distance_km(self, loc1, loc2):
        R = 6371
        lat1, lon1 = math.radians(loc1['lat']), math.radians(loc1['lon'])
        lat2, lon2 = math.radians(loc2['lat']), math.radians(loc2['lon'])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def generate(self):
        """Méthode principale pour générer tous les fichiers."""
        all_data = []
        print("Génération des lignes de données...")
        for _ in tqdm(range(self.config.DATASET_SIZE)):
            all_data.append(self._generate_single_row())

        self._write_csv(all_data)
        self._save_feature_config()
        print("\nTerminé !")
        print(f"-> Fichier de données '{self.config.CSV_FILENAME}' créé.")
        print(f"-> Fichier de configuration '{self.config.CONFIG_FILENAME}' créé.")

    def _generate_single_row(self):
        user = User()
        incident = Incident()
        
        distance = self._calculate_distance_km(
            user.get_current_location(incident.hour_of_day, incident.is_weekend),
            incident.location
        )
        urgency = (1 / (distance + 0.1)) * (1 / (incident.days_since + 1))

        geo, social, _, gravity = butterfly.get_feature_vector(user, incident, distance)
        is_useful = Oracle.decide_if_useful(incident, urgency, geo, social, gravity)
        
        # Construire la ligne de données
        row = {
            "distance_km": distance,
            "days_since_incident": incident.days_since,
            "hour_of_day": incident.hour_of_day,
            "user_engagement_rate": user.engagement_rate,
            "user_num_friends": len(user.friends),
            "user_connections": user.connections,
            "urgency_score": urgency,
            "user_type": user.user_type.value,
            "incident_type": incident.type.value,
            "day_of_week": incident.day_of_week,
            "is_weekend": 1 if incident.is_weekend else 0,
            "incident_zone_type": incident.zone_type,
            "is_useful": is_useful
        }
        return row

    def _write_csv(self, data):
        """Écrit les données dans un fichier CSV avec One-Hot Encoding."""
        header = self.numerical_features[:]
        for cat, values in self.categorical_features.items():
            if cat == "is_weekend": # Déjà binaire
                 header.append(cat)
            else:
                 header.extend([f"{cat}_{val}" for val in values])
        header.append("is_useful")

        filepath = self.config.OUTPUT_DIR / self.config.CSV_FILENAME
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for row_dict in tqdm(data, desc="Écriture du CSV"):
                row_list = [row_dict.get(feat) for feat in self.numerical_features]
                
                for cat, values in self.categorical_features.items():
                    if cat == "is_weekend":
                        row_list.append(row_dict.get(cat))
                    else:
                        for val in values:
                            is_present = 1 if row_dict.get(cat) == val else 0
                            row_list.append(is_present)
                
                row_list.append(row_dict.get("is_useful"))
                writer.writerow(row_list)

    def _save_feature_config(self):
        """Sauvegarde la configuration des colonnes pour le script d'entraînement."""
        config_data = {
            "numerical_features": self.numerical_features,
            "categorical_features_one_hot": [
                f"{cat}_{val}"
                for cat, values in self.categorical_features.items() if cat != "is_weekend"
                for val in values
            ] + ["is_weekend"]
        }
        filepath = self.config.OUTPUT_DIR / self.config.CONFIG_FILENAME
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=4)

# --- 5. EXÉCUTION ---
if __name__ == "__main__":
    config = Config()
    generator = DatasetGenerator(config)
    generator.generate()
