import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

# Collecte des données
# Remplacez le chemin ci-dessous par le chemin de votre propre fichier CSV contenant les données
data = pd.read_csv("chemin/vers/votre/fichier.csv")

# Nettoyage et prétraitement des données
data = data.drop_duplicates()
data = data.dropna()

numerical_features = ['surface_habitable', 'nombre_de_chambres', 'proximite_transports']
categorical_features = ['type_de_propriete', 'quartier']

# Exploration des données
sns.pairplot(data)
plt.show()

# Sélection des caractéristiques
X = data[numerical_features + categorical_features]
y = data['prix']

# Préparation des données pour l'entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du pipeline de traitement
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)])

# Construction du modèle
models = [
    ('linear', LinearRegression()),
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('random_forest', RandomForestRegressor())
]

for name, model in models:
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{name} - R2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

# Optimisation et réglage des hyperparamètres
# Ici, nous utilisons l'exemple d'un RandomForestRegressor
param_grid = {
    'model__n_estimators': [10, 50, 100, 200],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor())])
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
