import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow Version:", tf.__version__)

# --- ÉTAPE 1 : CHARGEMENT ET PRÉPARATION DES DONNÉES ---

# Charger le jeu de données
try:
    df = pd.read_csv('lisbon_neural_data.csv')
except FileNotFoundError:
    print("Erreur : Le fichier 'lisbon_neural_data.csv' n'a pas été trouvé.")
    print("Veuillez d'abord exécuter le script de génération de données.")
    exit()

print("Aperçu des données chargées :")
print(df.head())
print("\nColonnes du DataFrame :")
print(df.columns)

# Séparer les caractéristiques (X) de la cible (y)
X = df.drop('is_useful', axis=1)
y = df['is_useful']

# --- ÉTAPE 2 : PRÉ-TRAITEMENT (SCALING) ---

# Identifier les colonnes numériques à mettre à l'échelle
# IMPORTANT : On ne met pas à l'échelle les colonnes qui sont déjà en 0/1 (One-Hot Encoded)
numerical_cols = [
    'distance_km', 
    'days_since_incident', 
    'user_engagement_rate', 
    'user_num_friends', 
    'user_connections'
]
print(f"\nColonnes numériques à mettre à l'échelle : {numerical_cols}")

# Diviser les données en ensembles d'entraînement et de test AVANT le scaling
# pour éviter la fuite de données (data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialiser le scaler
scaler = StandardScaler()

# Adapter le scaler UNIQUEMENT sur les données d'entraînement et les transformer
X_train_scaled_numerical = scaler.fit_transform(X_train[numerical_cols])

# Utiliser le scaler DÉJÀ ADAPTÉ pour transformer les données de test
X_test_scaled_numerical = scaler.transform(X_test[numerical_cols])

# Créer des DataFrames avec les données mises à l'échelle
X_train_scaled_numerical_df = pd.DataFrame(X_train_scaled_numerical, columns=numerical_cols, index=X_train.index)
X_test_scaled_numerical_df = pd.DataFrame(X_test_scaled_numerical, columns=numerical_cols, index=X_test.index)

# Remplacer les colonnes numériques originales par les versions mises à l'échelle
X_train_processed = X_train.drop(columns=numerical_cols).join(X_train_scaled_numerical_df)
X_test_processed = X_test.drop(columns=numerical_cols).join(X_test_scaled_numerical_df)

print("\nAperçu des données traitées et prêtes pour l'entraînement :")
print(X_train_processed.head())


# --- ÉTAPE 3 : CONSTRUCTION DU MODÈLE DE RÉSEAU DE NEURONES ---

model = keras.Sequential([
    # Couche d'entrée : la forme correspond au nombre de caractéristiques
    keras.layers.Input(shape=(X_train_processed.shape[1],)),

    # Première couche cachée : 64 neurones, fonction d'activation ReLU
    keras.layers.Dense(64, activation='relu'),
    # Couche de Dropout pour éviter le sur-apprentissage (overfitting)
    keras.layers.Dropout(0.3),

    # Deuxième couche cachée : 32 neurones
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),

    # Couche de sortie : 1 neurone, fonction d'activation sigmoid car c'est une classification binaire (0 ou 1)
    keras.layers.Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Indispensable pour la classification binaire
    metrics=['accuracy']
)

model.summary()


# --- ÉTAPE 4 : ENTRAÎNEMENT DU MODÈLE ---

print("\n--- DÉBUT DE L'ENTRAÎNEMENT ---")
history = model.fit(
    X_train_processed, 
    y_train,
    epochs=20, # Le nombre de fois que le modèle voit l'ensemble des données
    batch_size=64, # La taille des lots de données pour chaque mise à jour des poids
    validation_split=0.1, # Utiliser 10% des données d'entraînement pour la validation
    verbose=1
)
print("--- FIN DE L'ENTRAÎNEMENT ---")


# --- ÉTAPE 5 : ÉVALUATION DU MODÈLE ---

print("\n--- ÉVALUATION SUR L'ENSEMBLE DE TEST ---")
loss, accuracy = model.evaluate(X_test_processed, y_test)
print(f"\nPerte (Loss) sur les données de test : {loss:.4f}")
print(f"Précision (Accuracy) sur les données de test : {accuracy:.4f}")

# Obtenir les prédictions
y_pred_proba = model.predict(X_test_processed).flatten() # Aplatir pour obtenir un array 1D
y_pred_class = (y_pred_proba > 0.5).astype(int) # Convertir les probabilités en classes (0 ou 1)

# Afficher le rapport de classification
print("\n--- RAPPORT DE CLASSIFICATION ---")
print(classification_report(y_test, y_pred_class, target_names=['Inutile (0)', 'Utile (1)']))

# Afficher la matrice de confusion
print("\n--- MATRICE DE CONFUSION ---")
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Inutile', 'Utile'], yticklabels=['Inutile', 'Utile'])
plt.xlabel('Prédiction')
plt.ylabel('Vraie valeur')
plt.title('Matrice de Confusion')
plt.show()

