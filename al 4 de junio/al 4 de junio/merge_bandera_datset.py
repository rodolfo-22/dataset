import pandas as pd

print("1. Cargando archivos...")
# Cargar tu dataset original (el grande)
df_original = pd.read_csv('dataset_preci_temp_ndwi_ndmi_final1.csv', sep=',', encoding='utf-8')

# Cargar el dataset de resultados que generó el script de DTW
df_banderas = pd.read_csv('dataset_clasificado_banderas.csv', sep=',', encoding='utf-8')

# Limpiar nombres de columnas por seguridad
df_original.columns = df_original.columns.str.strip()
df_banderas.columns = df_banderas.columns.str.strip()

print("2. Preparando llaves de cruce...")
# Asegurarnos de que el dataset original tenga la columna 'año' para poder hacer el cruce
df_original['fecha_real'] = pd.to_datetime(df_original['fecha'])
df_original['año'] = df_original['fecha_real'].dt.year

print("3. Fusionando datasets (Merge)...")
# Hacemos un "Left Join": Mantenemos todo el df_original y le pegamos la Bandera_Cultivo
# donde coincidan la latitud, longitud y el año.
df_final = pd.merge(
    df_original,
    df_banderas[['lat', 'lon', 'año', 'Bandera_Cultivo']], # Solo traemos lo que nos importa
    on=['lat', 'lon', 'año'],
    how='left'
)

# --- DETALLE DE LIMPIEZA ---
# Como el DTW solo lo corrimos para 'Cropland', los que eran 'Grassland' 
# o bosque quedarán vacíos (NaN) en la nueva columna. 
# Rellenamos esos vacíos usando el land_cover_type original para que tu columna quede perfecta.
df_final['Bandera_Cultivo'] = df_final['Bandera_Cultivo'].fillna(df_final['land_cover_type'])

# (Opcional) Borrar las columnas temporales de fecha que creamos para el cruce
df_final = df_final.drop(columns=['fecha_real', 'año'])

print("4. Guardando el dataset integrado...")
nombre_salida = 'dataset_original_CON_BANDERAS.csv'
df_final.to_csv(nombre_salida, index=False, sep=',', encoding='utf-8')

print("-" * 50)
print(f"¡Listo! El archivo '{nombre_salida}' tiene todas tus filas originales y la nueva columna integrada.")