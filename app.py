import cv2
import numpy as np
import streamlit as st

# 1. GABARITO OFICIAL (25 QUESTÕES)
GABARITO = {
    1: 'A', 2: 'C', 3: 'B', 4: 'A', 5: 'D', 6: 'D', 7: 'C', 8: 'B', 9: 'A', 10: 'C',
    11: 'A', 12: 'D', 13: 'C', 14: 'A', 15: 'B', 16: 'C', 17: 'B', 18: 'D', 19: 'C', 20: 'D',
    21: 'C', 22: 'D', 23: 'A', 24: 'D', 25: 'C'
}

OPCOES = ['A', 'B', 'C', 'D']

st.set_page_config(page_title="Leitor AvaliaBH", page_icon="📝", layout="centered")

st.title("📝 Leitor de Gabarito Automático")
st.caption("AvaliaBH / CAEd - M0701 - 7º Ano Matemática")

uploaded_file = st.camera_input("Tire uma foto clara do cartão de respostas")

if uploaded_file is not None:
    # Converter foto enviada para o formato OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    target_w, target_h = 1000, 1400
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Processamento e Binarização da imagem
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Busca pelas âncoras quadradas nos cantos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ancoras = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            area = cv2.contourArea(c)
            if 0.75 <= ar <= 1.25 and area > 300:
                ancoras.append((x, y, w, h, x + w/2, y + h/2))

    # Alinhamento por Perspectiva
    if len(ancoras) >= 4:
        ancoras = sorted(ancoras, key=lambda a: a[1]) # Ordena verticalmente
        topo = sorted(ancoras[:2], key=lambda a: a[0]) # 2 do topo
        base = sorted(ancoras[-2:], key=lambda a: a[0]) # 2 da base
        
        pts1 = np.float32([
            [topo[0][4], topo[0][5]],
            [topo[1][4], topo[1][5]],
            [base[0][4], base[0][5]],
            [base[1][4], base[1][5]]
        ])
        pts2 = np.float32([[0, 0], [target_w, 0], [0, target_h], [target_w, target_h]])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped_color = cv2.warpPerspective(img, M, (target_w, target_h))
        thresh_warped = cv2.warpPerspective(thresh, M, (target_w, target_h))
    else:
        warped_color = cv2.resize(img, (target_w, target_h))
        thresh_warped = cv2.resize(thresh, (target_w, target_h))

    img_display = warped_color.copy()

    # Mapeamento da grade de respostas (4 colunas)
    colunas_questoes = [
        (1, 7, 0.08, 0.28),    # Coluna 1: Q1 a Q7
        (8, 14, 0.31, 0.51),   # Coluna 2: Q8 a Q14
        (15, 21, 0.54, 0.74),  # Coluna 3: Q15 a Q21
        (22, 25, 0.77, 0.97)   # Coluna 4: Q22 a Q25
    ]

    y_grid_start, y_grid_end = 0.52, 0.92
    respostas_detectadas = {}
    acertos = 0

    # Leitura das bolinhas marcadas
    for q_start, q_end, x_start_rel, x_end_rel in colunas_questoes:
        for idx_q, q_num in enumerate(range(q_start, q_end + 1)):
            y_center = int((y_grid_start + (y_grid_end - y_grid_start) * (idx_q + 0.5) / 7.0) * target_h)
            
            pixel_counts = []
            coords_opcoes = []

            for idx_op in range(4):
                x_center = int((x_start_rel + (x_end_rel - x_start_rel) * (0.38 + idx_op * 0.19)) * target_w)
                r = 14
                
                x1, y1 = max(0, x_center - r), max(0, y_center - r)
                x2, y2 = min(target_w, x_center + r), min(target_h, y_center + r)
                
                roi = thresh_warped[y1:y2, x1:x2]
                count = cv2.countNonZero(roi) if roi.size > 0 else 0
                pixel_counts.append(count)
                coords_opcoes.append((x_center, y_center))

            max_val = max(pixel_counts)
            opcao_marcada = OPCOES[pixel_counts.index(max_val)] if max_val > 130 else None
            respostas_detectadas[q_num] = opcao_marcada

            gabarito_correto = GABARITO[q_num]
            
            # Marcação visual dos resultados na imagem
            for idx_op, (cx, cy) in enumerate(coords_opcoes):
                letra_op = OPCOES[idx_op]
                if opcao_marcada == letra_op:
                    if opcao_marcada == gabarito_correto:
                        cv2.circle(img_display, (cx, cy), 16, (0, 255, 0), 3) # Verde
                    else:
                        cv2.circle(img_display, (cx, cy), 16, (0, 0, 255), 3) # Vermelho
                elif letra_op == gabarito_correto and opcao_marcada != gabarito_correto:
                    cv2.circle(img_display, (cx, cy), 12, (255, 165, 0), 2) # Laranja (Correta)

            if opcao_marcada == gabarito_correto:
                acertos += 1

    nota = (acertos / 25.0) * 4.0

    # EXIBIÇÃO NO STREAMLIT
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Acertos", f"{acertos} / 25")
    with col2:
        st.metric("Nota Final (0 a 4,0)", f"{nota:.2f}")

    st.subheader("📷 Alinhamento e Leitura Visual")
    st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), caption="Círculos: Verde (Acerto), Vermelho (Erro), Laranja (Gabarito)", use_column_width=True)

    st.subheader("📋 Detalhamento")
    dados_tabela = []
    for q in range(1, 26):
        marcado = respostas_detectadas.get(q) or "Em branco"
        correto = GABARITO[q]
        status = "✅ Correto" if marcado == correto else "❌ Incorreto"
        dados_tabela.append({"Questão": f"Q{q:02d}", "Marcado": marcado, "Gabarito": correto, "Resultado": status})

    st.dataframe(dados_tabela, use_container_width=True)
