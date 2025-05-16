# 📄 CORRETOR DE GABARITOS (App Streamlit)

Aplicativo web para correção automática de gabaritos estilo vestibular, desenvolvido com **Python + OpenCV + Streamlit**.

Permite:
✅ Upload do gabarito base (sem marcações)  
✅ Marcação interativa das respostas corretas  
✅ Upload do gabarito respondido (foto)  
✅ Correção automática (acertos/erros)  
✅ Exportação em **CSV** e **PDF**  
✅ Totalmente responsivo para **uso em celular**

---

## 🚀 Como utilizar

### 1️⃣ Passo a passo no Streamlit Cloud
1. Clone ou fork este repositório.
2. Acesse: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Clique em **New App**.
4. Escolha este repositório.
5. Configure:
   - Main file: `app_gabarito.py`
   - Branch: `main`
6. Clique em **Deploy**.
7. Pronto! O app estará no ar.

### 2️⃣ Estrutura de arquivos:
```
/CORRETOR-DE-GABARITOS/
├── app_gabarito.py
├── requirements.txt
├── packages.txt
└── .streamlit/
    └── config.toml
```

---

## ✅ Requisitos Técnicos

### requirements.txt
```
streamlit==1.35.0
opencv-python-headless==4.9.0.80
numpy==1.26.4
pandas==2.2.2
pillow==10.3.0
fpdf2==2.7.8
streamlit-drawable-canvas==0.9.3
```

### packages.txt
```
libgl1
```

### .streamlit/config.toml
```
[server]
headless = true
port = $PORT
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#4CAF50"
backgroundColor = "#F5F5F5"
secondaryBackgroundColor = "#E0E0E0"
textColor = "#262730"
font = "sans serif"
```

---

## 📱 Funcionalidades

| Etapa | Descrição |
|--------|-----------|
| **1. Upload Gabarito Base** | Faz upload da folha em branco e permite marcar as respostas corretas no próprio app (via canvas). |
| **2. Upload do Gabarito Respondido** | O usuário faz upload da foto da folha preenchida. O app alinha, identifica marcações e compara com o gabarito base. |
| **3. Correção e Exportação** | Exibe o resultado (certo/errado) e permite exportar para CSV e PDF. |

---

## 🛠️ Tecnologias
- Streamlit
- OpenCV
- NumPy
- Pandas
- Pillow
- FPDF2
- Streamlit Drawable Canvas

---

## 👨‍💻 Desenvolvido por:
Jones Neto  
[github.com/jonesoute](https://github.com/jonesoute)
