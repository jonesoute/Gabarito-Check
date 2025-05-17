# 📄 CORRETOR DE GABARITOS - Nova Versão (Streamlit)

Aplicativo web para correção automática de gabaritos no estilo vestibular, com nova abordagem simplificada e eficiente. Desenvolvido em **Python + OpenCV + Streamlit**.

## ✅ O que esse app faz:
- Cadastro do gabarito base por tabela (sem clicar na imagem).
- Upload da foto do gabarito respondido.
- Detecção automática das bolhas preenchidas.
- Correção automática: mostra acertos e erros.
- Exporta resultados em PDF e CSV.
- Compatível com **celulares e desktops**.

---

## 🚀 Como usar no Streamlit Cloud
1. Faça fork deste repositório.
2. Acesse: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Clique em **"New App"**.
4. Selecione:
   - Main file: `app_gabarito.py`
   - Branch: `main`
5. Clique em **Deploy**.

---

## 📁 Estrutura do Projeto
```
/Gabarito-Check/
├── app_gabarito.py
├── requirements.txt
├── packages.txt
└── .streamlit/
    └── config.toml
```

---

## 📦 Requisitos

### requirements.txt
```
streamlit==1.35.0
opencv-python-headless==4.9.0.80
numpy==1.26.4
pandas==2.2.2
pillow==10.3.0
fpdf2==2.7.8
```

### packages.txt
```
libgl1
```

### .streamlit/config.toml
```
[server]
headless = true
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

## 🛠️ Tecnologias
- Streamlit
- OpenCV
- NumPy
- Pandas
- Pillow (PIL)
- FPDF2

---

## 👨‍💻 Desenvolvido por:
Jones Neto  
[github.com/jonesoute](https://github.com/jonesoute)
