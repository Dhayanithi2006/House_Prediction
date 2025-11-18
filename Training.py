import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
 
MODEL_OUT = 'models/pipeline.joblib'

os.makedirs('models', exist_ok=True)

df =pd.read_csv("data.csv")
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

if 'price' not in numeric_cols:
    raise ValueError("Expected 'price' column in numeric columns.")

X = df[numeric_cols].drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = X_train.columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

joblib.dump(model, MODEL_OUT)
print(f"Model saved to: {MODEL_OUT}")

predictions = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, predictions))
print("R²:", r2_score(y_test, predictions))