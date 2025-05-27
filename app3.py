import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image

# Chargement des fichiers
model_xgb = joblib.load("model_xgb.pkl")
model_cnn = tf.keras.models.load_model("model_cnn.h5")
# Chargement des fichiers
label_encoder = joblib.load("label_encoder.pkl")
xgb_classes = label_encoder.classes_  # 18 classes
cnn_classes = np.load("pokemon_cnn_classes.npy", allow_pickle=True)  # 15 classes

print("Nombre de classes XGBoost:", len(xgb_classes))  # Doit afficher 18
print("Nombre de classes CNN:", len(cnn_classes))      # Doit afficher 15
feature_columns = joblib.load("feature_columns.pkl")

# ----------- Fonctions -----------
def preprocess_image(img):
    img = img.resize((64, 64))  # taille d'entraînement du CNN
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image_type(img):
    processed = preprocess_image(img)
    probas_cnn = model_cnn.predict(processed, verbose=0)[0]  # Shape (15,)
    
    probas_extended = np.zeros(len(xgb_classes))  # Shape (18,)
    
    # Alignement des index
    for cnn_class, proba in zip(cnn_classes, probas_cnn):
        if cnn_class in xgb_classes:
            idx = np.where(xgb_classes == cnn_class)[0][0]
            probas_extended[idx] = proba
    
    return probas_extended  # Shape (18,)

def predict_stats_type(stats_dict):
    df = pd.DataFrame([stats_dict])
    df = df[feature_columns]
    probas = model_xgb.predict_proba(df)[0]
    return probas

def predict_fusion(stats_dict, img, alpha=0.5):
    xgb_proba = predict_stats_type(stats_dict)
    cnn_proba = predict_image_type(img)  # Déjà aligné
    
    return alpha * xgb_proba + (1 - alpha) * cnn_proba
# ----------- Interface -----------
st.title("🔮 Prédiction du type de Pokémon")

tab1, tab2 = st.tabs(["📊 Par Stats", "🖼️ Par Image"])

with tab1:
    st.header("Prédiction par statistiques")
    st.markdown("**Entrez les statistiques de base :**")

    base_stats = {}
    for stat in ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]:
        base_stats[stat] = st.number_input(stat, value=50, key=f"stat_{stat}")
    base_stats["Legendary"] = st.checkbox("Legendary ?", value=False, key="stat_Legendary")

    # Génération des features
    stats_input = {
        "HP": base_stats["HP"],
        "Attack": base_stats["Attack"],
        "Defense": base_stats["Defense"],
        "Sp. Atk": base_stats["Sp. Atk"],
        "Sp. Def": base_stats["Sp. Def"],
        "Speed": base_stats["Speed"],
        "Legendary": int(base_stats["Legendary"]),
        "IsDualType": 0,
        "Atk_Def_Ratio": base_stats["Attack"] / (base_stats["Defense"] + 1),
        "SpAtk_SpDef_Ratio": base_stats["Sp. Atk"] / (base_stats["Sp. Def"] + 1),
        "Physical_Total": base_stats["Attack"] + base_stats["Defense"],
        "Special_Total": base_stats["Sp. Atk"] + base_stats["Sp. Def"]
    }
    for col in feature_columns:
        if col not in stats_input:
            stats_input[col] = 0

    if st.button("Prédire (Stats)"):
        proba = predict_stats_type(stats_input)
        pred = xgb_classes[np.argmax(proba)]
        st.success(f"Type prédit : {pred}")

with tab2:
    st.header("Prédiction par image")
    img_file = st.file_uploader("Choisissez une image de Pokémon", type=["png", "jpg", "jpeg"])
    use_stats = st.checkbox("Combiner avec les stats ?")

    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Image chargée", use_column_width=True)

        if use_stats:
            st.subheader("Entrez les stats de base :")
            base_stats = {}
            for stat in ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]:
                base_stats[stat] = st.number_input(stat, value=50, key=f"fusion_{stat}")
            base_stats["Legendary"] = st.checkbox("Legendary ?", value=False, key="fusion_Legendary")

            stats_input = {
                "HP": base_stats["HP"],
                "Attack": base_stats["Attack"],
                "Defense": base_stats["Defense"],
                "Sp. Atk": base_stats["Sp. Atk"],
                "Sp. Def": base_stats["Sp. Def"],
                "Speed": base_stats["Speed"],
                "Legendary": int(base_stats["Legendary"]),
                "IsDualType": 0,
                "Atk_Def_Ratio": base_stats["Attack"] / (base_stats["Defense"] + 1),
                "SpAtk_SpDef_Ratio": base_stats["Sp. Atk"] / (base_stats["Sp. Def"] + 1),
                "Physical_Total": base_stats["Attack"] + base_stats["Defense"],
                "Special_Total": base_stats["Sp. Atk"] + base_stats["Sp. Def"]
            }
            for col in feature_columns:
                if col not in stats_input:
                    stats_input[col] = 0

            if st.button("Prédire (Fusion)"):
                try:
                    proba = predict_fusion(stats_input, img)
                    pred = xgb_classes[np.argmax(proba)]
                    st.success(f"Type prédit (fusionné) : {pred}")
                except ValueError as e:
                    st.error(f"Erreur de fusion : {str(e)}")

        else:
            if st.button("Prédire (Image)"):
                proba = predict_image_type(img)
                pred = xgb_classes[np.argmax(proba)]
                st.success(f"Type prédit : {pred}")
# Chargement des fichiers
label_encoder = joblib.load("label_encoder.pkl")
xgb_classes = label_encoder.classes_  # 18 classes
cnn_classes = np.load("pokemon_cnn_classes.npy", allow_pickle=True)  # 15 classes

# Vérifier les classes manquantes
missing_classes = set(xgb_classes) - set(cnn_classes)
print("Classes manquantes dans le CNN :", missing_classes)
print("Classes XGBoost:", xgb_classes)
print("Classes CNN:", cnn_classes)
