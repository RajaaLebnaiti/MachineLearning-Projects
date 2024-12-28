import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import random
import os
import face_recognition
# -------------------- Configurer la page --------------------
st.set_page_config(page_title="Suivi des Émotions des Étudiants", layout="wide")
st.title("🎓 Suivi en Temps Réel des Émotions des Étudiants")
st.markdown("**Analyse des émotions pour améliorer la concentration pendant les cours**")
st.markdown("---")

# -------------------- Définir les classes d'émotions --------------------
emotions = ['happy', 'nodesire', 'neutral', 'interest', 'surprised', 'fatigue']
colors = {
    "happy": "#00C853",  # Vert
    "nodesire": "#FF1744",  # Rouge
    "neutral": "#9E9E9E",  # Gris
    "interest": "#3D5AFE",  # Bleu
    "surprised": "#FFD600",  # Jaune
    "fatigue": "#6D4C41",  # Marron
}

# -------------------- Charger les images des étudiants --------------------
student_images_path = "students_images"
students_data = {}

if os.path.exists(student_images_path):
    for image_file in os.listdir(student_images_path):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # Charger l'image et extraire l'encodage facial
            image_path = os.path.join(student_images_path, image_file)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                student_name = os.path.splitext(image_file)[0]  # Extraire le nom sans l'extension
                students_data[student_name] = encoding[0]

# -------------------- Simuler des prédictions en temps réel --------------------
def simulate_predictions():
    # Simule les prédictions de classes pour 10 étudiants
    return [random.choice(emotions) for _ in range(10)]

# Initialiser un état pour les prédictions en temps réel
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "student_emotions" not in st.session_state:
    st.session_state.student_emotions = {}

# -------------------- Fonction pour calculer le taux de concentration --------------------
def concentration_rate(predictions):
    count_interest = predictions.count("interest")
    count_neutral = predictions.count("neutral")
    count_happy = predictions.count("happy")
    total = len(predictions)
    concentration_score = (count_interest + count_neutral + count_happy) / total if total > 0 else 0
    return concentration_score * 100

# -------------------- Fonction pour l'historique des émotions --------------------
def emotion_history(predictions):
    timestamps = [pd.Timestamp.now()] * len(predictions)
    history_df = pd.DataFrame({"timestamp": timestamps, "emotion": predictions})
    return history_df

# -------------------- Détection des émotions par étudiant --------------------
def detect_student_emotions(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(list(students_data.values()), face_encoding)
        name = "Inconnu"

        if True in matches:
            match_index = matches.index(True)
            name = list(students_data.keys())[match_index]

        if name not in st.session_state.student_emotions:
            st.session_state.student_emotions[name] = []

        # Ajouter une émotion simulée pour cet étudiant
        simulated_emotion = random.choice(emotions)
        st.session_state.student_emotions[name].append(simulated_emotion)






# -------------------- Vérification et Notification du Taux de Concentration --------------------
def check_concentration(concentration_score, threshold=50):
    if concentration_score < threshold:
        # Vérifier si le message n'a pas déjà été affiché
        if "low_concentration_message_shown" not in st.session_state:
            st.warning(f"⚠️ Taux de concentration faible ! Veuillez réengager les étudiants.")
            st.session_state.low_concentration_message_shown = True  # Marquer comme affiché
    else:
        # Si le taux de concentration est bon, réinitialiser l'état pour afficher de nouveau le message si nécessaire
        if "low_concentration_message_shown" in st.session_state:
            del st.session_state.low_concentration_message_shown





# -------------------- Historique des Émotions par Étudiant --------------------
def plot_student_emotion_history(student_name):
    if student_name in st.session_state.student_emotions:
        emotions = st.session_state.student_emotions[student_name]
        timestamps = [pd.Timestamp.now() for _ in range(len(emotions))]
        student_df = pd.DataFrame({"timestamp": timestamps, "emotion": emotions})
        st.line_chart(student_df.set_index("timestamp")["emotion"].value_counts().sort_index())
    else:
        st.info(f"Aucune donnée disponible pour l'étudiant {student_name}.")








# -------------------- Visualisation en Temps Réel --------------------


# Code principal de la caméra et du calcul du taux de concentration
st.header("🗊 Visualisation en Temps Réel")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📷 Vue en Direct de la Caméra")

    video_capture = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_clicked = st.button("🛑 Arrêter la Caméra", key="stop_camera")

    if video_capture.isOpened() and not stop_button_clicked:
        st.info("📺 Caméra en fonctionnement. Appuyez sur 'Arrêter la Caméra' pour stopper.")

        predictions_placeholder = st.empty()
        graph_placeholder = st.empty()
        concentration_placeholder = st.empty()
        history_placeholder = st.empty()
        student_graph_placeholder = st.empty()
        student_table_placeholder = st.empty()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.warning("⚠️ Échec de la capture d'image. Assurez-vous que la caméra est connectée.")
                break

            detect_student_emotions(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, caption="Flux en Direct", use_column_width=True)

            new_predictions = simulate_predictions()
            st.session_state.predictions.extend(new_predictions)

            predictions_df = pd.DataFrame(st.session_state.predictions, columns=["Émotion"])
            predictions_placeholder.dataframe(predictions_df.style.background_gradient(cmap="coolwarm"), height=250)

            emotion_counts = Counter(st.session_state.predictions)
            labels = emotion_counts.keys()
            sizes = emotion_counts.values()

            fig, ax = plt.subplots()
            ax.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=[colors[label] for label in labels],
                textprops={'fontsize': 12},
            )
            ax.axis('equal')
            graph_placeholder.pyplot(fig)

            concentration_score = concentration_rate(st.session_state.predictions)
            concentration_placeholder.markdown(f"### Taux de Concentration: {concentration_score:.2f}%")
            check_concentration(concentration_score)  # Vérification de la concentration

            history_df = emotion_history(st.session_state.predictions)
            history_placeholder.line_chart(history_df.set_index("timestamp")["emotion"].value_counts().sort_index())

            student_emotions_df = pd.DataFrame.from_dict(st.session_state.student_emotions, orient='index').transpose()
            student_table_placeholder.dataframe(student_emotions_df.fillna(""), height=250)

            student_emotion_counts = {name: Counter(emotions) for name, emotions in st.session_state.student_emotions.items()}
            student_graph_placeholder.bar_chart(pd.DataFrame(student_emotion_counts).fillna(0))

            if stop_button_clicked:
                break

    video_capture.release()

