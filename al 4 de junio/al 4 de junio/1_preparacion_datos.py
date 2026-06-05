# -*- coding: utf-8 -*-
"""
Script 1: Ingenieria de Caracteristicas y Control de Calidad Metodologico
Disenado para la investigacion del cultivo de maiz de primera en El Salvador.

VERSION OPTIMIZADA: Inclusion de Cropland y Grassland para mitigar el desbalance.
"""

import pandas as pd
import numpy as np
import os

def preparar_datos_sequia(ruta_input, ruta_output=None, encoding='utf-8'):
    print("=================================================================")
    print("   FASE 1: INGENIERIA DE CARACTERISTICAS Y CONTROL DE CALIDAD    ")
    print("=================================================================")
    print(f"[INFO] Cargando archivo CSV crudo: {ruta_input}...")
    
    try:
        df = pd.read_csv(ruta_input, encoding=encoding)
    except UnicodeDecodeError:
        print("[AVISO] Fallo lectura en UTF-8, intentando con codificacion 'latin1'...")
        df = pd.read_csv(ruta_input, encoding='latin1')

    columnas_requeridas = ['LST_Celsius', 'NDMI', 'precipitacion', 'fecha']
    for col in columnas_requeridas:
        if col not in df.columns:
            raise ValueError(f"[ERROR] La columna requerida '{col}' no existe en el archivo CSV.")

    # =========================================================================
    # ESCUDO DE FILTROS METODOLOGICOS (LIMPIEZA DE DATOS)
    # =========================================================================
    print("\n[PROCESO] Aplicando filtros metodologicos estrictos...")
    
    # A. Filtro de Uso de Suelo Flexible: Se incluye Cropland y Grassland para rescatar minifundios
    if 'land_cover_type' in df.columns:
        registros_antes = df.shape[0]
        # Limpiamos espacios en blanco y filtramos por ambas categorias
        df['land_cover_type'] = df['land_cover_type'].str.strip()
        df = df[df['land_cover_type'].isin(['Cropland', 'Grassland'])]
        registros_despues = df.shape[0]
        print(f" - Filtro de Suelo: Se conservaron {registros_despues} de {registros_antes} registros (Cropland + Grassland).")
    else:
        print("[AVISO] No se encontro la columna 'land_cover_type'. Se omitio el filtro de suelo.")
    
    # B. Filtro Temporal Fenologico: Limitar a la campana de primera (Junio, Julio, Agosto)
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.dropna(subset=['fecha'])
    
    df['mes_temp'] = df['fecha'].dt.month
    df = df[df['mes_temp'].isin([6, 7, 8])]
    print(f" - Filtro Cronologico: Dataset limitado a los meses de Junio (6), Julio (7) y Agosto (8).")
    
    df = df.drop(columns=['mes_temp'])

    if df.empty:
        raise ValueError("[ERROR] El dataset quedo completamente vacio despues de aplicar los filtros.")

    # =========================================================================
    # PROCESAMIENTO MATEMATICO E INGENIERIA DE VARIABLES
    # =========================================================================
    print("\n[PROCESO] Ejecutando normalizaciones Min-Max de variables fisicas...")
    
    min_lst, max_lst = df['LST_Celsius'].min(), df['LST_Celsius'].max()
    min_ndmi, max_ndmi = df['NDMI'].min(), df['NDMI'].max()
    min_precip, max_precip = df['precipitacion'].min(), df['precipitacion'].max()
    eps = 1e-6
    
    df['LST_norm'] = (df['LST_Celsius'] - min_lst) / (max_lst - min_lst + eps)
    df['NDMI_norm'] = (df['NDMI'] - min_ndmi) / (max_ndmi - min_ndmi + eps)
    df['Precip_norm'] = (df['precipitacion'] - min_precip) / (max_precip - min_precip + eps)

    print("[PROCESO] Formulando el Indice de Estres Termo-Hidrico (IETH)...")
    df['IETH'] = ((0.5 * df['LST_norm']) + (0.5 * (1 - df['NDMI_norm']))) * 100

    print("[PROCESO] Asignando umbrales de alerta institucionales...")
    condiciones = [
        (df['IETH'] >= 75),
        (df['IETH'] >= 55) & (df['IETH'] < 75),
        (df['IETH'] >= 35) & (df['IETH'] < 55),
        (df['IETH'] < 35)
    ]
    categorias = ['Sequia Extrema', 'Sequia Moderada', 'Estres Leve', 'Condicion Optima']
    df['IETH_Clasificacion'] = np.select(condiciones, categorias, default='No Determinado')

    df = df.drop(columns=['LST_norm', 'NDMI_norm', 'Precip_norm'])

    if ruta_output is None:
        nombre_base, ext = os.path.splitext(ruta_input)
        ruta_output = f"{nombre_base}_enriquecido{ext}"

    df.to_csv(ruta_output, index=False, encoding='utf-8')
    print(f"\n[PROCESO EXITOSO]")
    print(f" Matriz resultante depurada y enriquecida exportada en: {ruta_output}\n")
    return ruta_output

if __name__ == "__main__":
    archivo_csv_entrada = "dataset_maiz_santa_ana.csv" 
    
    try:
        preparar_datos_sequia(archivo_csv_entrada)
    except Exception as e:
        print(f"\nAnomalia en la ejecucion: {e}")