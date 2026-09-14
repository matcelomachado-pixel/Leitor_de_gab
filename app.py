import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# CONFIGURAÇÕES DA PROVA E GABARITO OFICIAL (25 QUESTÕES)
# ---------------------------------------------------------
GABARITO = {
    1: 'A', 2: 'C', 3: 'B', 4: 'A', 5: 'D', 6: 'D', 7: 'C', 8: 'B', 9: 'A', 10: 'C',
    11: 'A', 12: 'D', 13: 'C', 14: 'A', 15: 'B', 16: 'C', 17: 'B', 18: 'D', 19: 'C', 20: 'D',
    21: 'C', 22: 'D', 23: 'A', 24: 'D', 25: 'C'
}
TOTAL_QUESTOES = 25
NOTA_MAXIMA = 4.0
OPCOES = ['A', 'B', 'C', 'D']

# ---------------------------------------------------------
# FUNÇÕES DE ALINHAMENTO POR PONTOS DE ANCORAGEM
# ---------------------------------------------------------
def alinhar_perspectiva(image):
    """Localiza os 4 marcadores quadrados pretos de forma adaptativa e retifica a imagem."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Limiar Adaptativo ignora sombras e variações de luz
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 51, 15
    )

    # Encontra contornos dos marcadores pretos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centros = []
    h_img, w_img = gray.shape
    
    for c in contours:
        area = cv2.contourArea(c)
        # Ajuste na tolerância de tamanho dos quadrados
        if 150 < area < (h_img * w_img * 0.1):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True) # Margem maior para cantos arredondados
            
            if 4 <= len(approx) <= 6:  # Aceita formas próximas de quadrados
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centros.append([cX, cY])

    # Remove centros duplicados muito próximos uns dos outros
    centros_filtrados = []
    for pt in centros:
        if not centros_filtrados:
            centros_filtrados.append(pt)
        else:
            distancias = [np.linalg.norm(np.array(pt) - np.array(f)) for f in centros_filtrados]
            if min(distancias) > 50: # Se estiver a mais de 50 pixels de outro centro
                centros_filtrados.append(pt)

    # Se encontrar ao menos 4 marcadores, aplica a perspectiva
    if len(centros_filtrados) >= 4:
        centros_np = np.array(centros_filtrados, dtype="float32")
        
        # Encontra os 4 pontos mais extremos da imagem (os 4 cantos reais)
        s = centros_np.sum(axis=1)
        diff = np.diff(centros_np, axis=1)
        
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = centros_np[np.argmin(s)]       # Top-Left
        rect[2] = centros_np[np.argmax(s)]       # Bottom-Right
        rect[1] = centros_np[np.argmin(diff)]    # Top-Right
        rect[3] = centros_np[np.argmax(diff)]    # Bottom-Left
        
        width_dst, height_dst = 1000, 1400
        dst = np.array([
            [0, 0],
            [width_dst - 1, 0],
            [width_dst - 1, height_dst - 1],
            [0, height_dst - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width_dst, height_dst))
        return warped, True

    # Caso falhe completamente, retorna a imagem como está
    return cv2.resize(image, (1000, 1400)), False

# ---------------------------------------------------------
# LEITURA DAS BOLINHAS NA IMAGEM PADRONIZADA (1000x1400)
# ---------------------------------------------------------
def processar_folha_resposta(image_np):
    # Alinha a folha usando os cantos
    img_aligned, alinhado_sucesso = alinhar_perspectiva(image_np)
    
    gray = cv2.cvtColor(img_aligned, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    
    img_display = img_aligned.copy()
    respostas_lidas = {}
    
    # Coordenadas fixas (em pixels) para a imagem padronizada de 1000x1400
    colunas_blocos = [
        (1, 7, 100, 260),    # Q1 a Q7
        (8, 14, 330, 490),   # Q8 a Q14
        (15, 21, 560, 720),  # Q15 a Q21
        (22, 25, 790, 930)   # Q22 a Q25
    ]
    
    y_inicio = 790
    y_fim = 1260
    
    for q_start, q_end, x_min, x_max in colunas_blocos:
        num_q = (q_end - q_start) + 1
        
        for idx_q in range(num_q):
            q_num = q_start + idx_q
            if q_num > TOTAL_QUESTOES:
                break
                
            y1 = int(y_inicio + (idx_q / 7) * (y_fim - y_inicio))
            y2 = int(y_inicio + ((idx_q + 0.85) / 7) * (y_fim - y_inicio))
            
            pixels_opcoes = []
            coordenadas_opcoes = []
            
            for idx_opt in range(4):
                x1 = int(x_min + (idx_opt / 4) * (x_max - x_min))
                x2 = int(x_min + ((idx_opt + 1) / 4) * (x_max - x_min))
                
                roi = thresh[y1:y2, x1:x2]
                total_pixels = cv2.countNonZero(roi)
                pixels_opcoes.append(total_pixels)
                coordenadas_opcoes.append(((x1 + x2) // 2, (y1 + y2) // 2))
            
            max_pixels = max(pixels_opcoes)
            # Threshold de preenchimento
            if max_pixels > 120:
                opt_index = pixels_opcoes.index(max_pixels)
                respostas_lidas[q_num] = OPCOES[opt_index]
            else:
                respostas_lidas[q_num] = "N/A"
            
            # Feedback visual na tela
            gabarito_correto = GABARITO[q_num]
            for idx_opt, opt in enumerate(OPCOES):
                cx, cy = coordenadas_opcoes[idx_opt]
                if respostas_lidas[q_num] == opt:
                    if opt == gabarito_correto:
                        cv2.circle(img_display, (cx, cy), 12, (0, 255, 0), -1)
                    else:
                        cv2.circle(img_display, (cx, cy), 12, (255, 0, 0), -1)
                elif opt == gabarito_correto and respostas_lidas[q_num] != gabarito_correto:
                    cv2.circle(img_display, (cx, cy), 10, (255, 165, 0), 2)

    return respostas_lidas, img_display, alinhado_sucesso

# ---------------------------------------------------------
# INTERFACE DO USUÁRIO - STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="Leitor de Gabarito", layout="wide")
st.title("📋 Leitor Automático de Cartão-Resposta")

uploaded_file = st.file_uploader("Envie a foto do cartão-resposta", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    respostas_lidas, img_display, alinhado = processar_folha_resposta(image_np)
    
    if not alinhado:
        st.warning("⚠️ Marcadores de canto não foram detectados com clareza. A imagem foi apenas redimensionada. Recomenda-se tirar uma nova foto com melhor iluminação.")
    
    acertos = sum(1 for q, resp in respostas_lidas.items() if GABARITO.get(q) == resp)
    nota = round((acertos / TOTAL_QUESTOES) * NOTA_MAXIMA, 1)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Nota Final", f"{nota:.1f} / 4.0")
    col2.metric("Total de Acertos", f"{acertos} / {TOTAL_QUESTOES}")
    col3.metric("Aproveitamento", f"{(acertos / TOTAL_QUESTOES) * 100:.1f}%")
    
    st.divider()
    
    col_img, col_res = st.columns([1.2, 1])
    
    with col_img:
        st.subheader("Visualização da Leitura Alinhada")
        st.image(
            img_display,
            caption="Folha retificada via marcadores de canto",
            use_container_width=True
        )

    with col_res:
        st.subheader("Detalhamento por Questão")
        for q in range(1, TOTAL_QUESTOES + 1):
            resp_aluno = respostas_lidas.get(q, "N/A")
            gabarito_q = GABARITO.get(q)
            
            if resp_aluno == gabarito_q:
                st.markdown(f"**Questão {q:02d}:** :green[✔ {resp_aluno} (Correto)]")
            elif resp_aluno == "N/A":
                st.markdown(f"**Questão {q:02d}:** :orange[⚪ Em branco (Gabarito: {gabarito_q})]")
            else:
                st.markdown(f"**Questão {q:02d}:** :red[✖ Marcado: {resp_aluno} | Gabarito: {gabarito_q}]")
