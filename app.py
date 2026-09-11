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
# FUNÇÃO DE PROCESSAMENTO DE IMAGEM E LEITURA (OMR)
# ---------------------------------------------------------
def processar_folha_resposta(image_np):
    # Converte para escala de cinza e aplica binarização
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    img_display = image_np.copy()
    height, width = thresh.shape
    
    respostas_lidas = {}
    
    # Mapeamento aproximado das 4 colunas de blocos do cartão CAEd
    # Coluna 1: Q1-Q7 | Coluna 2: Q8-Q14 | Coluna 3: Q15-Q21 | Coluna 4: Q22-Q25
    colunas_blocos = [
        (1, 7, 0.08, 0.28),
        (8, 14, 0.31, 0.51),
        (15, 21, 0.54, 0.74),
        (22, 25, 0.77, 0.97)
    ]
    
    for q_start, q_end, x_min_pct, x_max_pct in colunas_blocos:
        num_q = (q_end - q_start) + 1
        y_top = int(height * 0.55)
        y_bottom = int(height * 0.92)
        
        for idx_q in range(num_q):
            q_num = q_start + idx_q
            if q_num > TOTAL_QUESTOES:
                break
                
            y1 = int(y_top + (idx_q / num_q) * (y_bottom - y_top))
            y2 = int(y_top + ((idx_q + 1) / num_q) * (y_bottom - y_top))
            
            pixels_opcoes = []
            coordenadas_opcoes = []
            
            for idx_opt, opt in enumerate(OPCOES):
                x1 = int(width * (x_min_pct + (idx_opt / 4) * (x_max_pct - x_min_pct)))
                x2 = int(width * (x_min_pct + ((idx_opt + 1) / 4) * (x_max_pct - x_min_pct)))
                
                roi = thresh[y1:y2, x1:x2]
                total_pixels = cv2.countNonZero(roi)
                pixels_opcoes.append(total_pixels)
                coordenadas_opcoes.append(((x1 + x2) // 2, (y1 + y2) // 2))
            
            # Identifica a opção mais preenchida
            max_pixels = max(pixels_opcoes)
            if max_pixels > 150:  # Limiar mínimo para considerar marcado
                opt_index = pixels_opcoes.index(max_pixels)
                respostas_lidas[q_num] = OPCOES[opt_index]
            else:
                respostas_lidas[q_num] = "N/A"  # Questão em branco
            
            # Desenha feedback visual na imagem
            gabarito_correto = GABARITO[q_num]
            for idx_opt, opt in enumerate(OPCOES):
                cx, cy = coordenadas_opcoes[idx_opt]
                if respostas_lidas[q_num] == opt:
                    if opt == gabarito_correto:
                        cv2.circle(img_display, (cx, cy), 12, (0, 255, 0), -1)  # Verde: Acerto
                    else:
                        cv2.circle(img_display, (cx, cy), 12, (255, 0, 0), -1)  # Vermelho: Erro
                elif opt == gabarito_correto and respostas_lidas[q_num] != gabarito_correto:
                    cv2.circle(img_display, (cx, cy), 10, (255, 165, 0), 2)  # Laranja: Resposta certa

    return respostas_lidas, img_display

# ---------------------------------------------------------
# INTERFACE DO USUÁRIO - STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="Leitor de Gabarito", layout="wide")
st.title("📋 Leitor Automático de Cartão-Resposta")

uploaded_file = st.file_uploader("Envie a foto do cartão-resposta", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Carrega imagem
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    # Processa leitura
    respostas_lidas, img_display = processar_folha_resposta(image_np)
    
    # Cálculo de acertos e nota proporcional (0 a 4 com 1 casa decimal)
    acertos = sum(1 for q, resp in respostas_lidas.items() if GABARITO.get(q) == resp)
    nota = round((acertos / TOTAL_QUESTOES) * NOTA_MAXIMA, 1)
    
    st.divider()
    
    # Exibição dos Resultados em Métricas Principais
    col1, col2, col3 = st.columns(3)
    col1.metric("Nota Final", f"{nota:.1f} / 4.0")
    col2.metric("Total de Acertos", f"{acertos} / {TOTAL_QUESTOES}")
    col3.metric("Aproveitamento", f"{(acertos / TOTAL_QUESTOES) * 100:.1f}%")
    
    st.divider()
    
    # Exibição Lado a Lado: Imagem Processada vs Tabela de Respostas
    col_img, col_res = st.columns([1.2, 1])
    
    with col_img:
        st.subheader("Visualização da Leitura")
        if img_display is not None:
            st.image(
                img_display,
                caption="Legenda: Verde (Acerto) | Vermelho (Erro) | Laranja (Gabarito Correto)",
                use_container_width=True
            )
        else:
            st.error("Erro ao carregar a visualização da imagem.")

    with col_res:
        st.subheader("Detalhamento por Questão")
        
        # Lista colorida formatada em Markdown
        for q in range(1, TOTAL_QUESTOES + 1):
            resp_aluno = respostas_lidas.get(q, "N/A")
            gabarito_q = GABARITO.get(q)
            
            if resp_aluno == gabarito_q:
                st.markdown(f"**Questão {q:02d}:** :green[✔ {resp_aluno} (Correto)]")
            elif resp_aluno == "N/A":
                st.markdown(f"**Questão {q:02d}:** :orange[⚪ Em branco (Gabarito: {gabarito_q})]")
            else:
                st.markdown(f"**Questão {q:02d}:** :red[✖ Marcado: {resp_aluno} | Gabarito: {gabarito_q}]")
