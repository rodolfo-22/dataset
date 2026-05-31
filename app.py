"""
Fase 3: Simulador Web Interactivo de Alertas Tempranas con Streamlit
Disenado para la investigacion del cultivo de maiz de primera en El Salvador.

Para ejecutar este script, utiliza el siguiente comando en tu terminal:
streamlit run app_interactiva.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

st.set_page_config(page_title="SAT - Sequia Agricola El Salvador", layout="wide")

@st.cache_resource
def entrenar_modelo_sat(ruta_csv):
    if not os.path.exists(ruta_csv):
        return None
    df = pd.read_csv(ruta_csv)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(by=['lon', 'lat', 'fecha']).reset_index(drop=True)
    df['IETH_siguiente_mes'] = df.groupby(['lon', 'lat'])['IETH'].shift(-1)
    df_modelo = df.dropna(subset=['IETH_siguiente_mes']).copy()
    df_modelo['mes_actual'] = df_modelo['fecha'].dt.month

    predictores = ['precipitacion', 'LST_Celsius', 'NDMI', 'NDWI', 'elevacion_msnm', 'mes_actual']
    target = 'IETH_siguiente_mes'
    
    X = df_modelo[predictores]
    y = df_modelo[target]
    
    modelo_rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=4, random_state=42, n_jobs=-1
    )
    modelo_rf.fit(X, y)
    return modelo_rf

# Inicializacion de la aplicacion
st.title("Sistema de Alerta Temprana de Sequia Agricola (SAT-SA)")
st.subheader("Modelado predictivo enfocado en el cultivo de maiz de primera (Santa Ana, El Salvador)")
st.write("Esta aplicacion interactiva permite simular escenarios hidroclimatologicos actuales para predecir el estres del proximo mes.")

archivo_datos = "dataset_maiz_santa_ana_enriquecido.csv"
modelo = entrenar_modelo_sat(archivo_datos)

if modelo is None:
    st.error(f"No se encontro el archivo '{archivo_datos}'. Por favor ejecuta primero el Script 1 para preparar los datos.")
else:
    # Diseno de la interfaz en dos columnas principales
    col_controles, col_resultados = st.columns([1, 1])
    
    with col_controles:
        st.header("Condiciones Hidroclimatologicas Actuales (Mes T)")
        
        mes_seleccionado = st.selectbox(
            "Mes en curso de la campana",
            options=[6, 7, 8],
            format_func=lambda x: {6: "Junio (Fase Vegetativa)", 7: "Julio (Floracion / Canicula)", 8: "Agosto (Maduracion)"}[x]
        )
        
        lst_celsius = st.slider("Temperatura Superficial de la Tierra (LST Celsius)", min_value=15.0, max_value=45.0, value=30.0, step=0.5)
        precipitacion = st.slider("Precipitacion acumulada del mes (mm)", min_value=0.0, max_value=400.0, value=120.0, step=5.0)
        ndmi = st.slider("Indice de Humedad Foliar (NDMI)", min_value=-0.50, max_value=0.50, value=0.10, step=0.01)
        ndwi = st.slider("Indice de Agua en Superficie (NDWI)", min_value=-0.50, max_value=0.50, value=-0.05, step=0.01)
        elevacion = st.number_input("Elevacion del terreno (msnm)", min_value=0, max_value=2500, value=500, step=50)

    with col_resultados:
        st.header("Pronostico Automatizado (Mes T+1)")
        
        # Construccion del vector predictor para la inferencia
        vector_entrada = pd.DataFrame([{
            'precipitacion': precipitacion,
            'LST_Celsius': lst_celsius,
            'NDMI': ndmi,
            'NDWI': ndwi,
            'elevacion_msnm': float(elevacion),
            'mes_actual': mes_seleccionado
        }])
        
        # Inferencia con el modelo entrenado
        ieth_proyectado = modelo.predict(vector_entrada)[0]
        
        # Determinacion del nivel de riesgo hídrico
        if ieth_proyectado >= 75:
            color_tarjeta = "#FFD2D2"
            color_texto = "#990000"
            dictamen = "ALERTA ROJA: Sequia Extrema. Alto riesgo de perdida del cultivo por estres termo-hidrico acumulado."
        elif ieth_proyectado >= 55:
            color_tarjeta = "#FFEAD2"
            color_texto = "#994C00"
            dictamen = "ALERTA NARANJA: Sequia Moderada. El rendimiento del grano experimentara reducciones significativas."
        elif ieth_proyectado >= 35:
            color_tarjeta = "#FFFFD2"
            color_texto = "#666600"
            dictamen = "ALERTA AMARILLA: Estres Leve. Condiciones de atencion preventiva para el desarrollo de la mazorca."
        else:
            color_tarjeta = "#D2FFD2"
            color_texto = "#006600"
            dictamen = "ALERTA VERDE: Condicion Optima. El equilibrio termo-hidrico es adecuado para la fenologia del cultivo."
            
        # Despliegue de los resultados metritos
        st.metric(label="Indice de Estres Termo-Hidrico (IETH) Proyectado", value=f"{ieth_proyectado:.2f} / 100")
        
        # Cuadro de dictamen institucional formateado con HTML/CSS nativo
        st.markdown(
            f"""
            <div style="background-color: {color_tarjeta}; padding: 20px; border-radius: 10px; border: 2px solid {color_texto};">
                <h3 style="color: {color_texto}; margin-top: 0px;">Dictamen del Sistema Predictivo</h3>
                <p style="color: {color_texto}; font-weight: bold; font-size: 16px;">{dictamen}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.write("")
        st.info("Nota metodologica: Este pronostico estima el nivel de estres que sufrira el suelo y la planta el proximo mes, aplicando un desfase temporal basado en el comportamiento historico registrado de la zona agricola.")