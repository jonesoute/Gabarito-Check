import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from fpdf import FPDF
import base64
import io
import os

st.set_page_config(page_title="App Gabarito", layout="centered")
st.title("📄 Correção Automática de Gabarito (IA + OpenCV)")

# Função para gerar PDF

def gerar_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    else:
        pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "Resultado da Correção", ln=True)
    pdf.ln(5)
    for _, row in df.iterrows():
        linha = f"Questão {row['Questão']}: Gabarito {row['Gabarito']} - Resposta {row['Resposta']} - {row['Resultado']}"
        pdf.cell(0, 8, linha, ln=True)
    return pdf.output(dest="S")

# Função para detectar ROI da folha e corrigir perspectiva
def detectar_e_corrigir_roi(imagem):
    img_resized = cv2.resize(imagem, (800, int(imagem.shape[0] * 800 / imagem.shape[1])))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnts = approx
            break
    else:
        return imagem  # se não detectar contorno, retorna original

    pts = doc_cnts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_resized, M, (maxWidth, maxHeight))
    return warped

# Função para detectar bolhas preenchidas automaticamente
def detectar_bolhas_auto(imagem, num_questoes):
    original = imagem.copy()
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bolhas = [c for c in contours if 300 < cv2.contourArea(c) < 2000]
    bolhas = sorted(bolhas, key=lambda c: cv2.boundingRect(c)[1])

    alternativas = ["A", "B", "C", "D", "E"]
    respostas = []

    por_questao = len(bolhas) // num_questoes
    for i in range(num_questoes):
        grupo = bolhas[i * por_questao:(i + 1) * por_questao]
        grupo = sorted(grupo, key=lambda c: cv2.boundingRect(c)[0])
        preenchimentos = []
        for c in grupo:
            x, y, w, h = cv2.boundingRect(c)
            roi = thresh[y:y + h, x:x + w]
            preenchimento = np.sum(roi == 255)
            preenchimentos.append((preenchimento, (x, y, w, h)))

        max_preench = max(p[0] for p in preenchimentos)
        marcadas = [alternativas[j] for j, (val, _) in enumerate(preenchimentos) if val >= max_preench * 0.6 and val > 100]
        if len(marcadas) == 1:
            resp = marcadas[0]
        elif len(marcadas) > 1:
            resp = "MULTIPLA"
        else:
            resp = "-"
        respostas.append({"questao": i + 1, "resposta": resp})

        # visual
        for j, (val, (x, y, w, h)) in enumerate(preenchimentos):
            cor = (0, 255, 0) if alternativas[j] in marcadas else (0, 0, 255)
            cv2.rectangle(original, (x, y), (x + w, y + h), cor, 1)
            cv2.putText(original, f"{i+1}{alternativas[j]}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)

    return respostas, original

# Interface
st.header("1️⃣ Parâmetros do Gabarito")
num_questoes = st.number_input("Número total de questões:", min_value=1, max_value=200, value=26)

# Tabela simples com gabarito correto
st.markdown("**Gabarito Correto:**")
gabarito_df = pd.DataFrame({"Questão": list(range(1, num_questoes + 1)), "Gabarito": ["A"] * num_questoes})
edited_gabarito = st.data_editor(gabarito_df, use_container_width=True)

# Upload
st.header("2️⃣ Enviar Gabarito Respondido")
img_file = st.file_uploader("Imagem do gabarito respondido:", type=["jpg", "jpeg", "png"])

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_raw is None:
        st.error("Erro ao carregar imagem.")
    else:
        st.image(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), caption="Imagem Original", use_container_width=True)
        img_alinhada = detectar_e_corrigir_roi(img_raw)
        st.image(cv2.cvtColor(img_alinhada, cv2.COLOR_BGR2RGB), caption="Imagem Alinhada", use_container_width=True)

        respostas_detectadas, img_anotada = detectar_bolhas_auto(img_alinhada, num_questoes)
        st.image(cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB), caption="Bolhas Detectadas", use_container_width=True)

        resultados = []
        for i in range(num_questoes):
            gabarito = edited_gabarito.iloc[i]["Gabarito"]
            resposta = respostas_detectadas[i]["resposta"] if i < len(respostas_detectadas) else "-"
            resultado = "CORRETA" if resposta == gabarito else "INCORRETA"
            if resposta == "MULTIPLA": resultado = "INCORRETA (Múltipla)"
            resultados.append({"Questão": i + 1, "Gabarito": gabarito, "Resposta": resposta, "Resultado": resultado})

        df_final = pd.DataFrame(resultados)
        st.subheader("📊 Resultado da Correção")
        st.dataframe(df_final, use_container_width=True)
        st.success(f"Total de acertos: {df_final['Resultado'].str.contains('CORRETA').sum()} / {num_questoes}")

        # Exportação
        st.header("3️⃣ Exportar Resultados")
        csv_bytes = df_final.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar CSV", csv_bytes, "resultado_gabarito.csv", "text/csv")

        pdf_bytes = gerar_pdf(df_final)
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        st.markdown(f"<a href='data:application/pdf;base64,{b64_pdf}' download='resultado_gabarito.pdf'>📥 Baixar PDF</a>", unsafe_allow_html=True)

        if st.button("🔄 Nova Correção"):
            st.experimental_rerun()
