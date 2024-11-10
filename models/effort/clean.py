import kagglehub
path = kagglehub.dataset_download("ajitjadhav1/strava-running-activity-data") + '/Strava Running Data.xlsx'

print("Path to dataset files:", path)

data = pd.read_excel(path)
# Filtrar solo actividades de tipo 'Run'
data = data[data['type'] == 'Run']

# Eliminar filas con valores nulos en las columnas clave, incluyendo 'total_elevation_gain'
data = data.dropna(subset=['distance', 'moving_time', 'average_speed', 'max_speed', 'total_elevation_gain'])

# Filtrar outliers básicos (distancia, tiempo, velocidad, y ganancia de elevación)
data = data[(data['distance'] > 0) & (data['moving_time'] > 0) & (data['average_speed'] > 0) & (data['max_speed'] > 0)]
data = data[(data['total_elevation_gain'] >= 0) & (data['total_elevation_gain'] < 1000)]  # Ajustar según el dataset

# Reiniciar índice después de la limpieza
data.reset_index(drop=True, inplace=True)

# Calcular esfuerzo con la ecuación ajustada
data['effort'] = ((data['distance'] / data['moving_time']) + 
                  ((data['average_speed'] + data['max_speed']) / 2)) * (1 + data['total_elevation_gain'] / 100)

# Conservar solo las columnas relevantes
data = data[['distance', 'moving_time', 'average_speed', 'max_speed', 'total_elevation_gain', 'effort']]

# Revisar el DataFrame final
data.head()