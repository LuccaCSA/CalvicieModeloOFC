import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
from sklearn.preprocessing import StandardScaler
import os

#caminho dos arquivos
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modeloCalvos.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

#carrega o modelo
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    return model

#carrega o scaler
@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

#processa a imagem
def preprocess_image(image, scaler):
    img = load_img(image, target_size=(64, 64), color_mode="grayscale")  # Redimensionar e converter para escala de cinza
    img_array = img_to_array(img).flatten().reshape(1, -1)  # Flatten e reshape para o formato do modelo
    img_array = scaler.transform(img_array)  # Normalizar os dados com o scaler treinado
    return img_array

#titulo
st.markdown(
    """
    <h1 style="text-align: center; color: #4CAF50;">
        🧑???CALVO ou CABELUDO???🧑
    </h1>
    """,
    unsafe_allow_html=True
)
#descrição
st.markdown(
    """
    <p style="text-align: center; font-size: 20px; color: #FFFFFF;">
        Descubra se o escalpo que você nao para de pensar é calvo ou nao!
        <br>Da sobrancelha p cima!!!
    </p>
    """,
    unsafe_allow_html=True,
)

#imagem
uploaded_file = st.file_uploader("Faça o upload de um escalpo(formatos suportados: PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    #mostrar a imagem carregada
    st.image(uploaded_file, caption="Imagem Carregada", use_column_width=True)

    #carregar modelo e scaler
    model = load_model()
    scaler = load_scaler()

    #preprocessar a imagem
    preprocessed_img = preprocess_image(uploaded_file, scaler)

    #fazer a previsão
    prediction = model.predict(preprocessed_img)
    probabilities = model.predict_proba(preprocessed_img)
    classes = ["Cabeludo", "Calvo"]

    #identificar a classe de maior confiança
    predicted_index = np.argmax(probabilities)  # Índice da classe prevista
    predicted_class = classes[predicted_index]  # Classe correspondente
    confidence = probabilities[0][predicted_index]  # Confiança da previsão

    #exibir os resultados
    st.subheader("Resultado da Classificação")
    st.write(f"Classe: {predicted_class}")
    st.write(f"Confiança: {confidence * 100:.2f}%")

    #calcular a porcentagem complementar
    complementary_percentage = (1 - confidence) * 100

    #mensagem condicional com base na classe prevista (nao canonica, apenas brincadeira)
    if predicted_class == "Cabeludo":
        st.markdown(
            f"""
            <p style="font-size: 18px; color: #FF4500;">
                Estima-se que o sujeito acima já perdeu <b>{complementary_percentage:.2f}%</b> do cabelo!
            </p>
            """,
            unsafe_allow_html=True
        )
    else:  # Caso seja "Calvo"
        st.markdown(
            f"""
            <p style="font-size: 18px; color: #FF4500;">
                Estima-se que o sujeito acima tem apenas <b>{complementary_percentage:.2f}%</b> de cabelo sobrando!
            </p>
            """,
            unsafe_allow_html=True
        )
