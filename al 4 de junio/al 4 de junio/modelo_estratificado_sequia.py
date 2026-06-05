# -*- coding: utf-8 -*-
"""
Sistema Inteligente de Alerta Temprana Multicultivo
Implementación de Machine Learning Estratificado (Random Forest)
Diseñado para predecir el Estrés Termo-Hídrico (IETH) en El Salvador.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def ejecutar_pipeline_multicultivo(ruta_input, ruta_output="predicciones_multicultivo.csv"):
    print("=================================================================")
    print("  FASE 1: INGENIERÍA DE CARACTERÍSTICAS Y CÁLCULO GLOBAL (IETH)  ")
    print("=================================================================")
    print(f"[INFO] Cargando dataset consolidado: {ruta_input}...")
    
    df = pd.read_csv(ruta_input, encoding='utf-8')
    df.columns = df.columns.str.strip()
    
    print("[PROCESO] Filtrando exclusivamente Granos Básicos y Caña de Azúcar...")
    cultivos_objetivo = ['Granos Básicos', 'Caña de Azúcar']
    df = df[df['Bandera_Cultivo'].isin(cultivos_objetivo)].copy()
    
    if df.empty:
        raise ValueError("[ERROR] El dataset quedó vacío tras aplicar el filtro de cultivos.")
    
    # 1. Asegurar formato temporal y crear variable de mes
    df['fecha_real'] = pd.to_datetime(df['fecha'])
    df['mes_actual'] = df['fecha_real'].dt.month
    
    # 2. Normalización Min-Max Global
    print("[PROCESO] Ejecutando normalizaciones Min-Max globales...")
    min_lst, max_lst = df['LST_Celsius'].min(), df['LST_Celsius'].max()
    min_ndmi, max_ndmi = df['NDMI'].min(), df['NDMI'].max()
    eps = 1e-6
    
    df['LST_norm'] = (df['LST_Celsius'] - min_lst) / (max_lst - min_lst + eps)
    df['NDMI_norm'] = (df['NDMI'] - min_ndmi) / (max_ndmi - min_ndmi + eps)
    
    # 3. Formulación del IETH
    print("[PROCESO] Formulando el Índice de Estrés Termo-Hídrico (IETH) para los 12 meses...")
    df['IETH'] = ((0.5 * df['LST_norm']) + (0.5 * (1 - df['NDMI_norm']))) * 100
    df = df.drop(columns=['LST_norm', 'NDMI_norm'])

    print("\n=================================================================")
    print("  FASE 2: ORDENAMIENTO ESPACIO-TEMPORAL Y TIME-LAGGING           ")
    print("=================================================================")
    # Ordenar estrictamente para que el shift() sea cronológico
    df = df.sort_values(by=['lon', 'lat', 'fecha_real']).reset_index(drop=True)
    
    print("[PROCESO] Aplicando desfase temporal (shift) para pronóstico...")
    # Cada fila obtiene el IETH que ocurrirá el próximo mes en esa misma coordenada
    df['IETH_siguiente_mes'] = df.groupby(['lon', 'lat'])['IETH'].shift(-1)
    
    # Eliminamos las filas ciegas (diciembre no tiene mes siguiente en el mismo año)
    df_modelo = df.dropna(subset=['IETH_siguiente_mes', 'Bandera_Cultivo']).copy()
    
    # Excluimos categorías no útiles
    df_modelo = df_modelo[df_modelo['Bandera_Cultivo'] != 'Indeterminado']

    print("\n=================================================================")
    print("  FASE 3: ENTRENAMIENTO ESTRATIFICADO POR TIPO DE CULTIVO        ")
    print("=================================================================")
    
    predictores = ['precipitacion', 'LST_Celsius', 'NDMI', 'NDWI', 'elevacion_msnm', 'mes_actual']
    target = 'IETH_siguiente_mes'
    
    cultivos_unicos = df_modelo['Bandera_Cultivo'].unique()
    diccionario_modelos = {}
    resultados_lista = []
    
    for cultivo in cultivos_unicos:
        print(f"\n--- ENTRENANDO MODELO ESPECIALIZADO: {cultivo.upper()} ---")
        
        # Aislar los datos exclusivos de este cultivo
        df_cultivo = df_modelo[df_modelo['Bandera_Cultivo'] == cultivo].copy()
        
        if len(df_cultivo) < 50:
            print(f"[AVISO] Insuficientes datos ({len(df_cultivo)}) para {cultivo}. Se omitirá.")
            continue
            
        X = df_cultivo[predictores]
        y = df_cultivo[target]
        
        # División 80% Entrenamiento / 20% Prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Configurar y entrenar el Random Forest
        modelo_rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=4, random_state=42, n_jobs=-1
        )
        modelo_rf.fit(X_train, y_train)
        
        # Guardar el modelo en el diccionario maestro
        diccionario_modelos[cultivo] = modelo_rf
        
        # Evaluación de Rendimiento
        y_pred = modelo_rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f" R² (Precisión): {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        
        # 5. Extracción de la Importancia de Variables
        importancias = modelo_rf.feature_importances_
        df_importancia = pd.DataFrame({
            'Variable': predictores, 
            'Peso': importancias
        }).sort_values(by='Peso', ascending=False)

        print(f"\n [ IMPORTANCIA DE VARIABLES PARA {cultivo.upper()} ]")
        for idx, fila in df_importancia.iterrows():
            print(f" - {fila['Variable']}: {fila['Peso']*100:.2f}%")
        print("-" * 60)
        
        # Generar las predicciones para todo el sub-dataset para exportarlas
        df_cultivo['IETH_Predicho_Proximo_Mes'] = modelo_rf.predict(X)
        resultados_lista.append(df_cultivo)

    print("\n=================================================================")
    print("  FASE 4: CONSOLIDACIÓN Y EXPORTACIÓN                            ")
    print("=================================================================")
    # Unir todos los sub-datasets predecidos en un solo archivo maestro
    df_final_predicciones = pd.concat(resultados_lista, ignore_index=True)
    df_final_predicciones.to_csv(ruta_output, index=False, encoding='utf-8')
    print(f"[ÉXITO] Archivo final de predicciones exportado en: {ruta_output}")
    
    return diccionario_modelos

def simulador_alerta_multicultivo(modelos_entrenados):
    print("\n=================================================================")
    print("  SIMULADOR DE ALERTAS TEMPRANAS (ESCENARIO HIPOTÉTICO)          ")
    print("=================================================================")
    
    # Escenario de Sequía Severa (Ejemplo: Julio sin lluvias, alta temperatura)
    datos_escenario = pd.DataFrame([{
        'precipitacion': 0.0,
        'LST_Celsius': 39.5,
        'NDMI': -0.15,
        'NDWI': -0.20,
        'elevacion_msnm': 500,
        'mes_actual': 7
    }])
    
    print("Condiciones actuales ingresadas (Mes 7): 0mm lluvia | 39.5°C | NDMI -0.15\n")
    
    # Evaluar el mismo escenario de clima en cada uno de los modelos especializados
    for cultivo, modelo in modelos_entrenados.items():
        prediccion_ieth = modelo.predict(datos_escenario)[0]
        
        # Asignación de Alertas Institucionales
        if prediccion_ieth >= 75: alerta = "🔴 ROJA (Sequía Extrema)"
        elif prediccion_ieth >= 55: alerta = "🟠 NARANJA (Sequía Moderada)"
        elif prediccion_ieth >= 35: alerta = "🟡 AMARILLA (Estrés Leve)"
        else: alerta = "🟢 VERDE (Condición Óptima)"
        
        print(f"Impacto proyectado en [{cultivo.upper()}]:")
        print(f" - IETH esperado mes próximo: {prediccion_ieth:.2f}/100 -> ALERTA {alerta}\n")
    print("=================================================================\n")

# Ejecución principal
if __name__ == "__main__":
    # Sustituye por el nombre del archivo que generamos en el paso del merge
    archivo_entrada = "dataset_original_CON_BANDERAS.csv" 
    
    try:
        modelos_dict = ejecutar_pipeline_multicultivo(archivo_entrada)
        if modelos_dict:
            simulador_alerta_multicultivo(modelos_dict)
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] Ocurrió un problema durante la ejecución: {e}")