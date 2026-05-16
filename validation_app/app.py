"""
Punto de entrada de la aplicación Streamlit.

Ejecutar desde la raíz del proyecto:
    uv run streamlit run validation_app/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="arXiv Classifier",
    page_icon="📄",
    layout="wide",
)

pg = st.navigation(
    [
        st.Page("pages/extraction.py",  title="Validación de extracción",    icon="🔍"),
        st.Page("pages/predictions.py", title="Validación de predicciones",  icon="🎯"),
        st.Page("pages/classifier.py",  title="Clasificar paper nuevo",      icon="🤖"),
    ]
)
pg.run()
