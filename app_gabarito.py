import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from fpdf import FPDF
import base64
import io

# Configuração de layout responsivo
st.set_page_config(page_title="App Gabarito", layout="centered")
st.markdown("""
<style>
img, canvas {
    max-width: 100% !important;
    height: auto !important;
}
.stCanvas > div {
    width: 100% !important;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 Correção Automática de Gabarito")

# Função para detectar orientação
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

# Gerar PDF com fpdf
def gerar_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Resultado da Correção", ln=True, align='C')
    pdf.ln(5)
    for _, row in df.iterrows():
        pdf.cell(0, 8, f"Questão {row['Questão']}: {row['Resposta']}", ln=True)
    return pdf.output(dest="S").encode("latin-1")

# Tela 1: Upload Gabarito Base
st.header("1️⃣ Upload do Gabarito Base")
num_questoes = st.number_input("Número de questões:", min_value=1, max_value=200, step=1)
uploaded_file = st.file_uploader("Selecione o gabarito base (imagem):", type=["jpg", "jpeg", "png"])

if 'coords' not in st.session_state:
    st.session_state.coords = []

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_corrigida = detectar_orientacao(img_bgr)
    pil_img = Image.fromarray(cv2.cvtColor(img_corrigida, cv2.COLOR_BGR2RGB))

    # Redimensiona imagem para largura máxima de 350px (mobile)
    canvas_width = min(pil_img.width, 350)
    canvas_height = int((canvas_width / pil_img.width) * pil_img.height)

    st.subheader("🖍️ Clique para marcar as respostas corretas:")
    canvas = st_canvas(
        fill_color="rgba(255,0,0,0.3)",
        stroke_width=5,
        background_image=pil_img,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="point",
        key="canvas1"
    )

    if canvas.json_data:
        objetos = canvas.json_data.get("objects", [])
        pontos = [(int(obj['left']), int(obj['top'])) for obj in objetos]
        st.session_state.coords = pontos[:num_questoes]

    st.info(f"Respostas marcadas: {len(st.session_state.coords)} / {num_questoes}")
    if st.button("🗑️ Limpar marcações"):
        st.session_state.coords = []

# Tela 2: Correção do Gabarito Respondido
if st.session_state.coords and len(st.session_state.coords) == num_questoes:
    st.header("2️⃣ Upload do Gabarito Respondido")
    resp_file = st.file_uploader("Selecione o gabarito preenchido:", type=["jpg", "jpeg", "png"], key="resp_gabarito")

    if resp_file:
        resp_bytes = np.asarray(bytearray(resp_file.read()), dtype=np.uint8)
        img_resp_bgr = cv2.imdecode(resp_bytes, cv2.IMREAD_COLOR)
        img_resp_corrigida = detectar_orientacao(img_resp_bgr)
        img_resp_gray = cv2.cvtColor(img_resp_corrigida, cv2.COLOR_BGR2GRAY)

        resultados = []
        for idx, (x, y) in enumerate(st.session_state.coords):
            size = 10
            y1, y2 = max(0, y - size), min(img_resp_gray.shape[0], y + size)
            x1, x2 = max(0, x - size), min(img_resp_gray.shape[1], x + size)
            crop = img_resp_gray[y1:y2, x1:x2]
            resultados.append("✅" if np.mean(crop) < 127 else "❌")

        df_resultados = pd.DataFrame({"Questão": list(range(1, num_questoes + 1)), "Resposta": resultados})
        st.subheader("📊 Resultados da Correção")
        st.dataframe(df_resultados, use_container_width=True)
        st.success(f"Acertos: {resultados.count('✅')} / {num_questoes}")

        st.image(cv2.cvtColor(img_resp_corrigida, cv2.COLOR_BGR2RGB), caption="Gabarito Respondido", use_column_width=True)

        # Tela 3: Exportação dos Resultados
        st.header("3️⃣ Exportar Resultados")
        csv_bytes = df_resultados.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar CSV", csv_bytes, "resultado_gabarito.csv", "text/csv")

        pdf_bytes = gerar_pdf(df_resultados)
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        st.markdown(f"<a href='data:application/pdf;base64,{b64_pdf}' download='resultado_gabarito.pdf'>📥 Baixar PDF</a>", unsafe_allow_html=True)

        if st.button("🔄 Nova Correção"):
            st.session_state.coords = []

