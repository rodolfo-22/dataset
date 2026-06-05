import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS (CSV)
# ==========================================
print("1. Cargando y preparando datos desde el CSV...")

# Asegúrate de poner el nombre exacto de tu archivo CSV
df_crudo = pd.read_csv('dataset_preci_temp_ndwi_ndmi_final1.csv', sep=',', encoding='utf-8')

# Limpiamos espacios en blanco invisibles en los nombres de las columnas
df_crudo.columns = df_crudo.columns.str.strip()

# Filtramos solo la tierra de cultivo usando el nombre de columna correcto
df_crudo = df_crudo[df_crudo['land_cover_type'] == 'Cropland'].copy()

# Convertimos la fecha (que ya viene en formato texto como YYYY-MM-DD) a tipo DateTime
df_crudo['fecha_real'] = pd.to_datetime(df_crudo['fecha'])

# Extraemos el mes y el año para la agrupación
df_crudo['mes'] = df_crudo['fecha_real'].dt.month
df_crudo['año'] = df_crudo['fecha_real'].dt.year

# ==========================================
# 2. AGRUPACIÓN (PIVOT) Y MANEJO DE HUECOS
# ==========================================
print("2. Agrupando datos por año y rellenando huecos (interpolación)...")

# Pivotar para crear una fila por coordenada y año, con los 12 meses como columnas
df_agrupado = pd.pivot_table(
    df_crudo, 
    values='NDVI', 
    index=['lat', 'lon', 'año', 'elevacion_msnm'], 
    columns=['mes'],
    aggfunc='mean'
).reset_index()

# Garantizar que existan las 12 columnas (Enero a Diciembre)
meses_necesarios = list(range(1, 13))
for m in meses_necesarios:
    if m not in df_agrupado.columns:
        df_agrupado[m] = np.nan

# Ordenar las columnas estructuralmente
cols_info = ['lat', 'lon', 'año', 'elevacion_msnm']
df_agrupado = df_agrupado[cols_info + meses_necesarios]

# Renombrar columnas para evitar confusiones (ndvi_m1, ndvi_m2, etc.)
meses_cols = [f'ndvi_m{m}' for m in meses_necesarios]
df_agrupado.columns = cols_info + meses_cols

# Interpolación lineal horizontal para rellenar meses sin datos (ej. nubes)
df_agrupado[meses_cols] = df_agrupado[meses_cols].interpolate(method='linear', axis=1, limit_direction='both')

# Eliminar filas que se hayan quedado completamente vacías en los 12 meses
df_agrupado = df_agrupado.dropna(subset=meses_cols, how='all')

# ==========================================
# 3. FIRMAS FENOLÓGICAS (CURVAS PATRÓN)
# ==========================================
print("3. Cargando las firmas de referencia...")

firmas_referencia = {
    'Caña de Azúcar': np.array([0.4, 0.4, 0.3, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.7, 0.3]),
    'Granos Básicos': np.array([0.2, 0.2, 0.2, 0.2, 0.4, 0.7, 0.8, 0.4, 0.7, 0.8, 0.4, 0.2]),
    'Cultivo Perenne (Café)': np.array([0.6, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7]),
    'Pastizal': np.array([0.3, 0.2, 0.2, 0.3, 0.5, 0.6, 0.7, 0.7, 0.7, 0.6, 0.5, 0.4])
}

# ==========================================
# 4. CLASIFICACIÓN CON DTW
# ==========================================
print("4. Evaluando similitudes temporales (aplicando DTW)...")

def clasificar_cultivo_dtw(serie_ndvi_pixel):
    # Candado de seguridad: Forzamos la serie a ser un arreglo de números decimales (float)
    serie_ndvi_pixel = np.array(serie_ndvi_pixel, dtype=float)
    
    # Si sigue habiendo algún nulo que no se pudo interpolar, se marca indeterminado
    if np.isnan(serie_ndvi_pixel).any():
        return 'Indeterminado'
    
    mejor_clase = None
    distancia_minima = float('inf')
    
    # Comparamos la curva del píxel actual contra las 4 firmas de referencia
    for clase, firma_patron in firmas_referencia.items():
        distancia, _ = fastdtw(serie_ndvi_pixel, firma_patron, dist=lambda a, b: abs(a - b))
        if distancia < distancia_minima:
            distancia_minima = distancia
            mejor_clase = clase
            
    return mejor_clase

# Candado de seguridad 2: Forzamos las columnas del DataFrame a ser float
df_agrupado[meses_cols] = df_agrupado[meses_cols].astype(float)

# Aplicar la función de DTW a lo largo de cada fila
df_agrupado['Bandera_Cultivo'] = df_agrupado[meses_cols].apply(
    lambda fila: clasificar_cultivo_dtw(fila.values),
    axis=1
)

# ==========================================
# 5. EXPORTACIÓN DE RESULTADOS
# ==========================================
print("5. Guardando resultados...")

# Exportar el resultado final a un nuevo archivo CSV
nombre_salida = 'dataset_clasificado_banderas.csv'
df_agrupado.to_csv(nombre_salida, index=False, sep=',', encoding='utf-8')

print("-" * 50)
print(f"¡Proceso terminado con éxito! El archivo '{nombre_salida}' ha sido creado.")
print("\n--- Vista previa de los resultados generados ---")
print(df_agrupado[['lat', 'lon', 'año', 'Bandera_Cultivo']].head(10))