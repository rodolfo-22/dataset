"""
Fase 2: Pronóstico Temporal (Time-Lagging) y Random Forest
Diseñado para la investigación del cultivo de maíz de primera en El Salvador.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def entrenar_pronostico_temporal(ruta_csv):
    print("=================================================================")
    print("      FASE 2: MACHINE LEARNING Y SISTEMA DE ALERTA TEMPRANA      ")
    print("=================================================================")
    
    if not os.path.exists(ruta_csv):
        print(f"Error: El archivo '{ruta_csv}' no existe.")
        return None

    df = pd.read_csv(ruta_csv)
    
    # 1. Estructuración Cronológica y Espacial
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(by=['lon', 'lat', 'fecha']).reset_index(drop=True)
    
    print("Aplicando desfase temporal (Time-Lagging) para pronóstico...")
    # IETH del próximo mes a partir de los datos de este mes
    df['IETH_siguiente_mes'] = df.groupby(['lon', 'lat'])['IETH'].shift(-1)
    df_modelo = df.dropna(subset=['IETH_siguiente_mes']).copy()
    df_modelo['mes_actual'] = df_modelo['fecha'].dt.month

    # 2. Configuración de Variables
    predictores = ['precipitacion', 'LST_Celsius', 'NDMI', 'NDWI', 'elevacion_msnm', 'mes_actual']
    target = 'IETH_siguiente_mes'
    
    X = df_modelo[predictores]
    y = df_modelo[target]

    # 3. Entrenamiento (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Entrenando Random Forest Regressor...")
    modelo_rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=4, random_state=42, n_jobs=-1
    )
    modelo_rf.fit(X_train, y_train)

    # 4. Métricas de Rendimiento
    y_pred = modelo_rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n[ MÈTRICAS DE RENDIMIENTO REAL - SIN FUGA DE DATOS ]")
    print(f" R² (Coeficiente de Determinación): {r2:.4f}")
    print(f" MAE (Error Absoluto Medio):        {mae:.4f}")
    print(f" RMSE (Raíz Error Cuadrático Medio):{rmse:.4f}\n")

    # 5. Importancia de Variables
    importancias = modelo_rf.feature_importances_
    df_importancia = pd.DataFrame({
        'Variable': predictores, 'Peso': importancias
    }).sort_values(by='Peso', ascending=False)

    print("[ IMPORTANCIA DE LAS VARIABLES PREDICTORAS ]")
    for idx, fila in df_importancia.iterrows():
        print(f" - {fila['Variable']}: {fila['Peso']*100:.2f}%")
        
    return modelo_rf

if __name__ == "__main__":
    archivo_enriquecido = "dataset_maiz_santa_ana_enriquecido.csv"
    
    modelo_rf = entrenar_pronostico_temporal(archivo_enriquecido)
    
    if modelo_rf is not None:
        print("\n=================================================================")
        print("      SIMULADOR DE ALERTAS TEMPRANAS (ESCENARIO HIPOTÉTICO)      ")
        print("=================================================================")
        
        escenario_extremo = pd.DataFrame([{
            'precipitacion': 0.0,         
            'LST_Celsius': 39.5,          
            'NDMI': -0.15,                
            'NDWI': -0.20,                
            'elevacion_msnm': 500,        
            'mes_actual': 7               
        }])

        prediccion_ieth = modelo_rf.predict(escenario_extremo)[0]
        
        if prediccion_ieth >= 75: alerta = "🔴 ALERTA ROJA: Sequía Extrema"
        elif prediccion_ieth >= 55: alerta = "🟠 ALERTA NARANJA: Sequía Moderada"
        elif prediccion_ieth >= 35: alerta = "🟡 ALERTA AMARILLA: Estrés Leve"
        else: alerta = "🟢 ALERTA VERDE: Condición Óptima"

        print("Condiciones ingresadas al modelo (Julio): 0 mm lluvia | 39.5°C | NDMI -0.15")
        print(f"PRONÓSTICO PARA EL PRÓXIMO MES (Agosto):")
        print(f" - IETH Proyectado: {prediccion_ieth:.2f} / 100")
        print(f" - Dictamen: {alerta}")
        print("=================================================================\n")