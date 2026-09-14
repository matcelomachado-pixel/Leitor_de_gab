def alinhar_perspectiva(image):
    """Localiza os 4 marcadores quadrados pretos de forma adaptativa e retifica a imagem."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # MUDANÇA AQUI: O Limiar Adaptativo ignora sombras e variações de luz
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
