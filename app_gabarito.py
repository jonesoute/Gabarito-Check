import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from fpdf import FPDF
import base64
import io

# Configurações iniciais
st.set_page_config(page_title="App Gabarito", layout="centered")
st.title("📄 Correção Automática de Gabarito - Nova Versão")

# Função para detectar orientação da imagem
def detectar_orientacao(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    altura, largura = imagem.shape[:2]
    centro_img = (largura // 2, altura // 2)
    marcadores = [(x + w // 2, y + h // 2) for c in contours if 100 < cv2.contourArea(c) < 5000 for x, y, w, h in [cv2.boundingRect(c)]]
    if len(marcadores) < 2: return imagem
    tl = sum(cx < centro_img[0] and cy < centro_img[1] for cx, cy in marcadores)
    bl = sum(cx < centro_img[0] and cy > centro_img[1] for cx, cy in marcadores)
    tr = sum(cx > centro_img[0] and cy < centro_img[1] for cx, cy in marcadores)
    br = sum(cx > centro_img[0] and cy > centro_img[1] for cx, cy in marcadores)
    if bl + br >= 2: return cv2.rotate(imagem, cv2.ROTATE_180)
    elif tr + br >= 2: return cv2.rotate(imagem, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif tl + bl >= 2: return cv2.rotate(imagem, cv2.ROTATE_90_CLOCKWISE)
    return imagem

# Função para gerar PDF
def gerar_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Resultado da Correção", ln=True, align='C')
    pdf.ln(5)
    for _, row in df.iterrows():
        pdf.cell(0, 8, f"Questão {row['Questão']}: Gabarito {row['Gabarito']} - Resposta {row['Resposta']} - {row['Resultado']}", ln=True)
    return pdf.output(dest="S").encode("latin-1")

# Etapa 1: Cadastro do Gabarito Base via Tabela
st.header("1️⃣ Defina o Gabarito Base (Tabela)")
num_questoes = st.number_input("Número total de questões:", min_value=1, max_value=200, step=1, value=10)

gabarito_base = []
for i in range(1, num_questoes + 1):
    alternativa = st.selectbox(f"Questão {i} - Alternativa correta:", ["A", "B", "C", "D", "E"], key=f"q{i}")
    gabarito_base.append(alternativa)

# Etapa 2: Upload do gabarito respondido
st.header("2️⃣ Upload do Gabarito Respondido")
resp_file = st.file_uploader("Selecione a imagem do gabarito respondido:", type=["jpg", "jpeg", "png"])

if resp_file:
    file_bytes = np.asarray(bytearray(resp_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_corrigida = detectar_orientacao(img_bgr)
    img_gray = cv2.cvtColor(img_corrigida, cv2.COLOR_BGR2GRAY)

    st.image(cv2.cvtColor(img_corrigida, cv2.COLOR_BGR2RGB), caption="Gabarito Respondido", use_column_width=True)

    st.info("🔍 Detectando bolhas preenchidas automaticamente...")

    # Simulação da detecção automática (exemplo: respostas marcadas aleatórias)
    alternativas = ["A", "B", "C", "D", "E"]
    respostas_marcadas = [np.random.choice(alternativas) for _ in range(num_questoes)]  # (Substituir pela detecção real)

    # Comparar com gabarito base
    resultados = []
    for i in range(num_questoes):
        if respostas_marcadas[i] == gabarito_base[i]:
            resultados.append("✅ Correto")
        else:
            resultados.append("❌ Errado")

    df_resultados = pd.DataFrame({
        "Questão": list(range(1, num_questoes + 1)),
        "Gabarito": gabarito_base,
        "Resposta": respostas_marcadas,
        "Resultado": resultados
    })

    st.subheader("📊 Resultado da Correção")
    st.dataframe(df_resultados, use_container_width=True)

    st.success(f"Total de acertos: {resultados.count('✅ Correto')} / {num_questoes}")

    # Etapa 3: Exportar Resultados
    st.header("3️⃣ Exportar Resultados")
    csv_bytes = df_resultados.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar CSV", csv_bytes, "resultado_gabarito.csv", "text/csv")

    pdf_bytes = gerar_pdf(df_resultados)
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    st.markdown(f"<a href='data:application/pdf;base64,{b64_pdf}' download='resultado_gabarito.pdf'>📥 Baixar PDF</a>", unsafe_allow_html=True)

    if st.button("🔄 Nova Correção"):
        st.experimental_rerun()
