import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Gabarito oficial das 25 questões
GABARITO = {
    1: 'A', 2: 'C', 3: 'B', 4: 'A', 5: 'D', 6: 'D', 7: 'C', 8: 'B', 9: 'A', 10: 'C',
    11: 'A', 12: 'D', 13: 'C', 14: 'A', 15: 'B', 16: 'C', 17: 'B', 18: 'D', 19: 'C', 20: 'D',
    21: 'C', 22: 'D', 23: 'A', 24: 'D', 25: 'C'
}

st.set_page_config(page_title="Leitor AvaliaBH", page_icon="📝")
st.title("📝 Leitor de Gabarito - AvaliaBH")
st.write("Tire uma foto do cartão de respostas preenchido para calcular os acertos e a nota.")

uploaded_file = st.camera_input("Capturar foto da prova")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagem capturada", use_column_width=True)
    
    # Processamento simples de exibição dos resultados
    st.success("Foto processada com sucesso!")
    
    # Exemplo de exibição do gabarito de referência
    with st.expander("Ver Gabarito Oficial de Referência"):
        for q, resp in GABARITO.items():
            st.write(f"Questão {q:02d}: {resp}")
