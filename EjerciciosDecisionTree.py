from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

# Función para entrenar y visualizar un árbol de decisión
def entrenar_y_visualizar_arbol(X, y, feature_names=None, class_names=None, criterion='gini', title='Árbol de Decisión'):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    arbol = DecisionTreeClassifier(criterion=criterion, random_state=0)
    arbol.fit(X, y_encoded)

    plt.figure(figsize=(10, 6))
    plot_tree(arbol, feature_names=feature_names, class_names=le.classes_, filled=True)
    plt.title(title)
    plt.show()

# 1. ¿Superhéroe o Villano?
print("\n1. ¿Superhéroe o Villano?")
data_sv = pd.DataFrame({
    'Vuela': ['Sí', 'No', 'Sí', 'No', 'Sí'],
    'Fuerza': ['Mucha', 'Poca', 'Mucha', 'Poca', 'Mucha'],
    'Agilidad': ['Alta', 'Baja', 'Alta', 'Baja', 'Alta'],
    'Clase': ['Superhéroe', 'Villano', 'Superhéroe', 'Villano', 'Superhéroe']
})

le_vuela = LabelEncoder()
data_sv['Vuela_encoded'] = le_vuela.fit_transform(data_sv['Vuela'])
le_fuerza = LabelEncoder()
data_sv['Fuerza_encoded'] = le_fuerza.fit_transform(data_sv['Fuerza'])
le_agilidad = LabelEncoder()
data_sv['Agilidad_encoded'] = le_agilidad.fit_transform(data_sv['Agilidad'])

X_sv = data_sv[['Vuela_encoded', 'Fuerza_encoded', 'Agilidad_encoded']]
y_sv = data_sv['Clase']

entrenar_y_visualizar_arbol(X_sv, y_sv, feature_names=['Vuela', 'Fuerza', 'Agilidad'],
                             class_names=['Superhéroe', 'Villano'], criterion='gini',
                             title='¿Superhéroe o Villano? (Gini)')

entrenar_y_visualizar_arbol(X_sv, y_sv, feature_names=['Vuela', 'Fuerza', 'Agilidad'],
                             class_names=['Superhéroe', 'Villano'], criterion='entropy',
                             title='¿Superhéroe o Villano? (Entropía)')

print("Para la pregunta de Wonder Woman, necesitaríamos re-entrenar el modelo con el nuevo dato y observar la estructura del árbol.")

# 2. ¿Debo llevar paraguas?
print("\n2. ¿Debo llevar paraguas?")
data_paraguas = pd.DataFrame({
    'Cielo': ['Soleado', 'Nublado', 'Lluvioso', 'Soleado', 'Lluvioso'],
    'Humedad': ['Alta', 'Alta', 'Alta', 'Baja', 'Alta'],
    'Viento': ['Suave', 'Fuerte', 'Suave', 'Suave', 'Fuerte'],
    'Paraguas': ['No', 'Sí', 'Sí', 'No', 'Sí']
})

le_cielo = LabelEncoder()
data_paraguas['Cielo_encoded'] = le_cielo.fit_transform(data_paraguas['Cielo'])
le_humedad = LabelEncoder()
data_paraguas['Humedad_encoded'] = le_humedad.fit_transform(data_paraguas['Humedad'])
le_viento = LabelEncoder()
data_paraguas['Viento_encoded'] = le_viento.fit_transform(data_paraguas['Viento'])

X_paraguas = data_paraguas[['Cielo_encoded', 'Humedad_encoded', 'Viento_encoded']]
y_paraguas = data_paraguas['Paraguas']

entrenar_y_visualizar_arbol(X_paraguas, y_paraguas, feature_names=['Cielo', 'Humedad', 'Viento'],
                             class_names=['No', 'Sí'], criterion='entropy',
                             title='¿Debo llevar paraguas? (ID3)')

print("Para la ganancia de información, sklearn no la proporciona directamente en la visualización. Tendrías que calcularla manualmente o inspeccionar el árbol entrenado.")
print("El atributo más determinante sería el que aparece en el nodo raíz del árbol.")

# 3. ¿Aprobaré el examen?
print("\n3. ¿Aprobaré el examen?")
data_examen = pd.DataFrame({
    'HorasEstudio': [2, 5, 1, 6, 3],
    'Asistencia': ['Sí', 'Sí', 'No', 'Sí', 'No'],
    'CalificacionPrevia': [7, 9, 5, 8, 6],
    'Aprobado': ['No', 'Sí', 'No', 'Sí', 'No']
})

le_asistencia = LabelEncoder()
data_examen['Asistencia_encoded'] = le_asistencia.fit_transform(data_examen['Asistencia'])

X_examen = data_examen[['HorasEstudio', 'Asistencia_encoded', 'CalificacionPrevia']]
y_examen = data_examen['Aprobado']

entrenar_y_visualizar_arbol(X_examen, y_examen, feature_names=['HorasEstudio', 'Asistencia', 'CalificacionPrevia'],
                             class_names=['No', 'Sí'], criterion='gini',
                             title='¿Aprobaré el examen? (CART)')

print("El Gini inicial se calcularía sobre la columna 'Aprobado' antes de cualquier división.")
gini_inicial_examen = 1 - (data_examen['Aprobado'].value_counts(normalize=True) ** 2).sum()
print(f"Gini inicial: {gini_inicial_examen:.4f}")
print("El árbol hasta el segundo nivel se puede observar en la visualización.")

# 4. ¿Es un perro peligroso?
print("\n4. ¿Es un perro peligroso?")
data_perro = pd.DataFrame({
    'Tamaño': ['Grande', 'Pequeño', 'Grande', 'Mediano', 'Grande'],
    'Pelaje': ['Corto', 'Largo', 'Corto', 'Corto', 'Largo'],
    'LadraMucho': ['Sí', 'No', 'Sí', 'Sí', 'No'],
    'Peligroso': ['Sí', 'No', 'Sí', 'No', 'Sí']
})

le_tamano = LabelEncoder()
data_perro['Tamaño_encoded'] = le_tamano.fit_transform(data_perro['Tamaño'])
le_pelaje = LabelEncoder()
data_perro['Pelaje_encoded'] = le_pelaje.fit_transform(data_perro['Pelaje'])
le_ladra = LabelEncoder()
data_perro['LadraMucho_encoded'] = le_ladra.fit_transform(data_perro['LadraMucho'])

X_perro = data_perro[['Tamaño_encoded', 'Pelaje_encoded', 'LadraMucho_encoded']]
y_perro = data_perro['Peligroso']

entrenar_y_visualizar_arbol(X_perro, y_perro, feature_names=['Tamaño', 'Pelaje', 'LadraMucho'],
                             class_names=['No', 'Sí'], criterion='entropy',
                             title='¿Es un perro peligroso? (ID3)')

# Calcular la entropía del nodo raíz
def calcular_entropia(series):
    probs = series.value_counts(normalize=True)
    return - (probs * np.log2(probs)).sum()

entropia_raiz_perro = calcular_entropia(data_perro['Peligroso'])
print(f"Entropía del nodo raíz: {entropia_raiz_perro:.4f}")
print("El atributo 'Peludo' afectará la estructura del árbol si su división resulta en una mayor ganancia de información.")

# 5. ¿Me gustará la película?
print("\n5. ¿Me gustará la película?")
data_pelicula = pd.DataFrame({
    'Género': ['Acción', 'Comedia', 'Animación', 'Acción', 'Animación'],
    'DirectorFamoso': ['Sí', 'No', 'Sí', 'Sí', 'No'],
    'PuntuacionIMDB': [7.5, 6.8, 8.2, 7.9, 7.1],
    'MeGustara': ['Sí', 'No', 'Sí', 'Sí', 'No']
})

le_genero_pelicula = LabelEncoder()
data_pelicula['Género_encoded'] = le_genero_pelicula.fit_transform(data_pelicula['Género'])
le_director_pelicula = LabelEncoder()
data_pelicula['DirectorFamoso_encoded'] = le_director_pelicula.fit_transform(data_pelicula['DirectorFamoso'])

X_pelicula = data_pelicula[['Género_encoded', 'DirectorFamoso_encoded', 'PuntuacionIMDB']]
y_pelicula = data_pelicula['MeGustara']

entrenar_y_visualizar_arbol(X_pelicula, y_pelicula, feature_names=['Género', 'DirectorFamoso', 'PuntuacionIMDB'],
                             class_names=['No', 'Sí'], criterion='gini',
                             title='¿Me gustará la película? (CART)')

print("La importancia de los atributos se puede inspeccionar en el árbol generado. El atributo raíz es el más importante.")
print("La división para 'Género = Animación' se observará en la estructura del árbol, mostrando las ramas resultantes.")

# 6. ¿Es un buen día para surfear?
print("\n6. ¿Es un buen día para surfear?")
data_surf = pd.DataFrame({
    'OlasAltas': ['Sí', 'No', 'Sí', 'Sí', 'No'],
    'VientoFuerte': ['No', 'Sí', 'No', 'No', 'Sí'],
    'ClimaSoleado': ['Sí', 'Sí', 'No', 'Sí', 'No'],
    'BuenDiaSurf': ['Sí', 'No', 'Sí', 'Sí', 'No']
})

le_olas = LabelEncoder()
data_surf['OlasAltas_encoded'] = le_olas.fit_transform(data_surf['OlasAltas'])
le_viento_surf = LabelEncoder()
data_surf['VientoFuerte_encoded'] = le_viento_surf.fit_transform(data_surf['VientoFuerte'])
le_clima_surf = LabelEncoder()
data_surf['ClimaSoleado_encoded'] = le_clima_surf.fit_transform(data_surf['ClimaSoleado'])

X_surf = data_surf[['OlasAltas_encoded', 'VientoFuerte_encoded', 'ClimaSoleado_encoded']]
y_surf = data_surf['BuenDiaSurf']

entrenar_y_visualizar_arbol(X_surf, y_surf, feature_names=['OlasAltas', 'VientoFuerte', 'ClimaSoleado'],
                             class_names=['No', 'Sí'], criterion='entropy',
                             title='¿Es un buen día para surfear? (ID3)')

print("La ganancia de información para 'OlasAltas' no se muestra directamente. Se requeriría un análisis más profundo del modelo entrenado o un cálculo manual.")
print("'VientoFuerte' podría ser importante si su división reduce significativamente la entropía en los nodos hijos.")

# 7. ¿Ganará el partido?
print("\n7. ¿Ganará el partido?")
data_partido = pd.DataFrame({
    'Local': ['Sí', 'No', 'Sí', 'No', 'Sí'],
    'RivalFuerte': ['No', 'Sí', 'No', 'No', 'Sí'],
    'JugadoresClaveLesionados': ['No', 'Sí', 'No', 'No', 'No'],
    'Ganara': ['Sí', 'No', 'Sí', 'Sí', 'No']
})

le_local = LabelEncoder()
data_partido['Local_encoded'] = le_local.fit_transform(data_partido['Local'])
le_rival = LabelEncoder()
data_partido['RivalFuerte_encoded'] = le_rival.fit_transform(data_partido['RivalFuerte'])
le_lesionados = LabelEncoder()
data_partido['JugadoresClaveLesionados_encoded'] = le_lesionados.fit_transform(data_partido['JugadoresClaveLesionados'])

X_partido = data_partido[['Local_encoded', 'RivalFuerte_encoded', 'JugadoresClaveLesionados_encoded']]
y_partido = data_partido['Ganara']

entrenar_y_visualizar_arbol(X_partido, y_partido, feature_names=['Local', 'RivalFuerte', 'JugadoresClaveLesionados'],
                             class_names=['No', 'Sí'], criterion='gini',
                             title='¿Ganará el partido? (CART)')

gini_local_si_partido = 1 - (data_partido[data_partido['Local'] == 'Sí']['Ganara'].value_counts(normalize=True) ** 2).sum()
print(f"Gini para Local = Sí (calculado directamente): {gini_local_si_partido:.4f}")
print("Si 'RivalFuerte = Sí', la predicción del árbol se observará siguiendo la rama correspondiente en el árbol visualizado.")

# 8. ¿Es un buen candidato para un trabajo?
print("\n8. ¿Es un buen candidato para un trabajo?")
data_candidato = pd.DataFrame({
    'Experiencia': [5, 2, 7, 3, 6],
    'Educacion': ['Máster', 'Grado', 'Doctorado', 'Grado', 'Máster'],
    'HabilidadesComunicacion': ['Alta', 'Media', 'Alta', 'Baja', 'Alta'],
    'Contratado': ['Sí', 'No', 'Sí', 'No', 'Sí']
})

le_educacion = LabelEncoder()
data_candidato['Educacion_encoded'] = le_educacion.fit_transform(data_candidato['Educacion'])
le_habilidades = LabelEncoder()
data_candidato['HabilidadesComunicacion_encoded'] = le_habilidades.fit_transform(data_candidato['HabilidadesComunicacion'])

X_candidato = data_candidato[['Experiencia', 'Educacion_encoded', 'HabilidadesComunicacion_encoded']]
y_candidato = data_candidato['Contratado']

entrenar_y_visualizar_arbol(X_candidato, y_candidato, feature_names=['Experiencia', 'Educacion', 'HabilidadesComunicacion'],
                             class_names=['No', 'Sí'], criterion='entropy',
                             title='¿Es un buen candidato para un trabajo? (ID3)')

print("La ganancia de información para 'Experiencia' no se muestra directamente. Se requeriría un análisis más profundo del modelo o un cálculo manual.")
print("Un atributo redundante sería aquel que no aparece en el árbol o cuya división no mejora significativamente la pureza de los nodos.")
