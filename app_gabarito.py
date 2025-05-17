import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from fpdf import FPDF
import base64
import io

# Configuração da página
st.set_page_config(page_title="App Gabarito", layout="centered")
st.title("📄 Correção Automática de Gabarito - Versão Colunas")

# Função para detectar orientação da imagem e rotacionar para horizontal
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

# Função para detectar bolhas preenchidas por colunas
def detectar_respostas_por_coluna(imagem, num_questoes):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    height, width = thresh.shape
    respostas = []

    # Definir número de colunas (automático para 4 colunas)
    colunas = 4
    questoes_por_coluna = num_questoes // colunas
    alternativas = ["A", "B", "C", "D", "E"]

    for i in range(colunas):
        x_ini = int((width / colunas) * i)
        x_fim = int((width / colunas) * (i + 1))
        coluna = thresh[:, x_ini:x_fim]
        alt_w = coluna.shape[1] // 5
        alt_h = coluna.shape[0] // questoes_por_coluna

        for q in range(questoes_por_coluna):
            questao_idx = i * questoes_por_coluna + q + 1
            preenchida = ""
            for a in range(5):
                x1 = a * alt_w
                y1 = q * alt_h
                bolha = coluna[y1:y1 + alt_h, x1:x1 + alt_w]
                media = np.mean(bolha)
                if media < 127:
                    if preenchida:
                        preenchida = "MULTIPLA"  # marcação dupla
                    else:
                        preenchida = alternativas[a]
            respostas.append({"questao": questao_idx, "resposta": preenchida if preenchida else "-"})

    return respostas

# Etapa 1: Cadastro do Gabarito Base
st.header("1️⃣ Defina o Gabarito Base (Tabela)")
num_questoes = st.number_input("Número total de questões:", min_value=1, max_value=200, step=1, value=26)

gabarito_base = []
for i in range(1, num_questoes + 1):
    alternativa = st.selectbox(f"Questão {i} - Alternativa correta:", ["A", "B", "C", "D", "E"], key=f"q{i}")
    gabarito_base.append(alternativa)

# Etapa 2: Upload do gabarito respondido
df_resultados = None
st.header("2️⃣ Upload do Gabarito Respondido")
resp_file = st.file_uploader("Selecione a imagem do gabarito respondido:", type=["jpg", "jpeg", "png"])

if resp_file:
    file_bytes = np.asarray(bytearray(resp_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_corrigida = detectar_orientacao(img_bgr)

    st.image(cv2.cvtColor(img_corrigida, cv2.COLOR_BGR2RGB), caption="Gabarito Respondido", use_column_width=True)
    st.info("🔍 Detectando bolhas preenchidas...")

    respostas_detectadas = detectar_respostas_por_coluna(img_corrigida, num_questoes)

    # Comparação com gabarito base
    resultados = []
    for i in range(num_questoes):
        questao = i + 1
        resp = respostas_detectadas[i]["resposta"]
        certo = gabarito_base[i]
        resultado = "✅" if resp == certo else "❌"
        if resp == "MULTIPLA": resultado = "❌ (Múltipla)"
        resultados.append({"Questão": questao, "Gabarito": certo, "Resposta": resp, "Resultado": resultado})

    df_resultados = pd.DataFrame(resultados)
    st.subheader("📊 Resultado da Correção")
    st.dataframe(df_resultados, use_container_width=True)
    st.success(f"Total de acertos: {df_resultados['Resultado'].str.contains('✅').sum()} / {num_questoes}")

# Etapa 3: Exportação
if df_resultados is not None:
    st.header("3️⃣ Exportar Resultados")
    csv_bytes = df_resultados.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar CSV", csv_bytes, "resultado_gabarito.csv", "text/csv")

    pdf_bytes = gerar_pdf(df_resultados)
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    st.markdown(f"<a href='data:application/pdf;base64,{b64_pdf}' download='resultado_gabarito.pdf'>📥 Baixar PDF</a>", unsafe_allow_html=True)

    if st.button("🔄 Nova Correção"):
        st.experimental_rerun()
