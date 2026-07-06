import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# URL RAW del dataset
url = "AQUI_PEGA_LA_URL_RAW_DEL_CSV"

# Cargar dataset
df = pd.read_csv(url)

# Limpieza de datos
df = df.drop_duplicates()

# Filtrado de valores extremos
df = df[df["selling_price"] < df["selling_price"].quantile(0.99)]
df = df[df["km_driven"] < df["km_driven"].quantile(0.99)]

# Crear variable marca desde el nombre del vehículo
df["brand"] = df["name"].str.split().str[0]

# Variables predictoras y variable objetivo
X = df[["year", "km_driven", "fuel", "seller_type", "transmission", "owner", "brand"]]
y = df["selling_price"]

# División de datos: 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Variables numéricas y categóricas
numeric_features = ["year", "km_driven"]
categorical_features = ["fuel", "seller_type", "transmission", "owner", "brand"]

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Modelo Ridge seleccionado
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ]
)

# Entrenar modelo
model.fit(X_train, y_train)

# Predicción en prueba
y_pred = model.predict(X_test)

# Métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Evaluación del modelo Ridge")
print("RMSE:", rmse)
print("R2:", r2)

# Crear carpeta models si no existe
os.makedirs("models", exist_ok=True)

# Guardar modelo entrenado
joblib.dump(model, "models/modelo_ridge_pipeline.pkl")

print("Modelo guardado en models/modelo_ridge_pipeline.pkl")
