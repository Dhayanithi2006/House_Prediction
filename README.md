# House_Prediction
A complete end-to-end Machine Learning + Flask project that predicts the price of a house based on multiple property features. The project includes data preprocessing, model training, model deployment, and a clean web interface for real-time predictions.

🚀 Project Overview

This project uses a regression model trained on housing data to estimate property prices. It includes:

Full ML pipeline (preprocessing + model)

Flask-based web application

Real-time predictions from user input

Modular project structure with model/, app/, and data/

🛠️ Features

🔧 Preprocessing Pipeline

One-hot encoding for categorical features

Scaling numerical features

Automatic transformation using ColumnTransformer

🤖 Machine Learning Model

RandomForestRegressor (or your model)

Saved as pipeline.joblib

🌐 Web App

Built with Flask

Clean form-based UI

Instant price prediction

project/ │ ├── model/ │ ├── pipeline.joblib │ └── train_model.py │ ├── app/ │ ├── predict_app.py

│ ├── data/ │ └── dataset.csv │ └── README.md

📦 Installation & Setup 1️⃣ Clone the Repository git clone <repo_link> cd project

2️⃣ Create & Activate Virtual Environment python -m venv venv venv\Scripts\activate

3️⃣ Install Dependencies pip install -r requirements.txt

4️⃣ Run the Flask App cd app python predict_app.py

🧠 Model Training

To retrain the model, run:

python model/train_model.py

this will open an website:Local URL: http://localhost:8501/

This will:

Load the dataset

Preprocess columns

Train the ML model

Save the full pipeline as pipeline.joblib
