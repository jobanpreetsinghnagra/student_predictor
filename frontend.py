import pandas as pd
import numpy as np
import gradio as gr
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and preprocess data
df = pd.read_csv("dataset/Student_Performance.csv")
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'No': 0, 'Yes': 1})

# Separate features and label
X = df.drop("Performance Index", axis=1).to_numpy()
y = df["Performance Index"].to_numpy()

# Split data 
#Train 60% , CV 20% , Test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# Scale input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train model
model = SGDRegressor(max_iter=1000, loss='squared_error', random_state=42)
model.fit(X_train_scaled, y_train)




def predict_performance(hours_studied, previous_score, extracurricular, sleep_hours, sample_papers):
    # Construct input array
    input_array = np.array([[hours_studied, previous_score, extracurricular, sleep_hours, sample_papers]])

    # Scale using the same scaler as training
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Optional: clamp to realistic range
    prediction = np.clip(prediction, 0, 100)

    return f"Predicted Performance Index: {round(prediction, 2)}"



interface = gr.Interface(
    fn=predict_performance,
    inputs=[
        gr.Slider(0, 10, step=0.1, label="Hours Studied"),
        gr.Slider(0, 100, step=1, label="Previous Scores"),
        gr.Radio(["No", "Yes"], label="Extracurricular Activities", type="index"),
        gr.Slider(0, 12, step=0.5, label="Sleep Hours"),
        gr.Slider(0, 50, step=1, label="Sample Question Papers Practiced"),
    ],
    outputs="text",
    title="Student Performance Predictor",
    description="Enter your study habits to predict your academic performance (0–100 scale).",
)

interface.launch(inline=True)
