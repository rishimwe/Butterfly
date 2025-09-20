import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.utils import class_weight
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json # MODIFIÉ : Import nécessaire
import joblib # MODIFIÉ : Import nécessaire pour la sauvegarde

print("TensorFlow Version:", tf.__version__)

# --- ÉTAPE 1 : CHARGEMENT DYNAMIQUE DES DONNÉES ---
# MODIFIÉ : Cette section charge maintenant les fichiers générés par le script "expert"

try:
    df = pd.read_csv('lisbon_expert_data.csv')
    with open('feature_config.json', 'r') as f:
        feature_config = json.load(f)
    print("Données et configuration chargées avec succès depuis les fichiers 'expert'.")
except FileNotFoundError:
    print("Erreur : Assurez-vous d'avoir exécuté 'generate_expert_dataset.py' avant ce script.")
    print("Les fichiers 'lisbon_expert_data.csv' et 'feature_config.json' sont nécessaires.")
    exit()

print("\n--- Analyse du déséquilibre des classes ---")
print(df['is_useful'].value_counts(normalize=True))
print("-----------------------------------------")

X = df.drop('is_useful', axis=1)
y = df['is_useful']

# --- ÉTAPE 2 : PRÉ-TRAITEMENT AUTOMATISÉ ---

# MODIFIÉ : La liste des colonnes numériques est lue depuis le fichier de config, plus de hardcoding !
numerical_cols = feature_config['numerical_features']
print(f"\nColonnes numériques à scaler (depuis config) : {numerical_cols}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
# Important : On utilise .copy() pour éviter les avertissements de Pandas
X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])


class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print(f"\nPoids calculés pour les classes : {class_weights_dict}")


# --- ÉTAPE 3 : CONSTRUCTION DU MODÈLE EXPERT (Inchangé) ---

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(128, kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(32, kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

model.summary()


# --- ÉTAPE 4 : ENTRAÎNEMENT AVEC DES CALLBACKS AVANCÉS (Inchangé) ---

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

print("\n--- DÉBUT DE L'ENTRAÎNEMENT AVANCÉ ---")
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping_callback, learning_rate_scheduler],
    class_weight=class_weights_dict,
    verbose=1
)
print("--- FIN DE L'ENTRAÎNEMENT ---")


# --- ÉTAPE 5 : ÉVALUATION MULTI-SEUILS (Inchangé) ---

print("\n--- ÉVALUATION FINALE SUR L'ENSEMBLE DE TEST ---")
y_pred_proba = model.predict(X_test).flatten()
THRESHOLDS = [0.5, 0.7, 0.8, 0.9]

for threshold in THRESHOLDS:
    print("\n" + "="*50)
    print(f"   RAPPORT DE CLASSIFICATION POUR UN SEUIL DE {threshold}")
    print("="*50)
    
    y_pred_class = (y_pred_proba > threshold).astype(int)
    print(classification_report(y_test, y_pred_class, target_names=['Inutile (0)', 'Utile (1)']))
    
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Inutile', 'Utile'], yticklabels=['Inutile', 'Utile'])
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie valeur')
    plt.title(f'Matrice de Confusion (Seuil = {threshold})')
    plt.show()


# --- ÉTAPE 6 : VISUALISATIONS D'EXPERT (Inchangé) ---

print("\n" + "="*50)
print("   VISUALISATIONS AVANCÉES POUR LE CHOIX DU SEUIL")
print("="*50)

fig, ax = plt.subplots(figsize=(10, 8))
PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba, ax=ax, name="Modèle Expert")
ax.set_title("Courbe Précision-Rappel")
ax.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax, name="Modèle Expert")
ax.set_title("Courbe ROC (Receiver Operating Characteristic)")
ax.grid(True)
plt.show()


# --- NOUVEAU : ÉTAPE 7 : SAUVEGARDE DES ARTEFACTS POUR LA PRODUCTION ---

print("\n--- SAUVEGARDE DES ARTEFACTS POUR LA PRÉDICTION ---")

model.save('butterfly_expert_model.keras')
print("Modèle expert sauvegardé dans 'butterfly_expert_model.keras'")

joblib.dump(scaler, 'expert_scaler.joblib')
print("Scaler expert sauvegardé dans 'expert_scaler.joblib'")

model_columns = list(X_train.columns)
with open('expert_model_columns.json', 'w') as f:
    json.dump(model_columns, f)
print("Ordre des colonnes sauvegardé dans 'expert_model_columns.json'")
