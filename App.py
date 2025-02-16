import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.losses import MeanSquaredError

# Map the loss function "mse" to Keras' built-in MeanSquaredError
custom_objects = {"mse": MeanSquaredError()}

# Load improved models
rnn_model = load_model("RNN_model.h5", custom_objects=custom_objects)
lstm_model = load_model("LSTM_model.h5", custom_objects=custom_objects)
gru_model = load_model("GRU_model.h5", custom_objects=custom_objects)

# Function to create input sequences
def create_sequences(data_x, time_steps=10):
    X = []
    for i in range(len(data_x) - time_steps):
        X.append(data_x[i:i+time_steps])
    return np.array(X)

# Streamlit app
st.title("Efficient Edge: Energy Forecasting for IIoT-Enabled Data Servers")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file with required columns", type=["csv"])

if uploaded_file:
    # Load uploaded CSV file
    df = pd.read_csv(uploaded_file, parse_dates=['DATETIME'], index_col='DATETIME')
    st.write("### Uploaded Data Sample:")
    st.write(df.head())

    # Check required columns
    required_columns = ['voltaje', 'corriente', 'frecuencia', 'energia', 'fp', 'potencia']
    if not all(col in df.columns for col in required_columns):
        st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
    else:
        # Scale data
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        scaled_features = scaler_x.fit_transform(df[['voltaje', 'corriente', 'frecuencia', 'energia', 'fp']])
        scaled_target = scaler_y.fit_transform(df[['potencia']])

        # Create input sequences
        time_steps = 10
        X = create_sequences(scaled_features, time_steps)
        true_values = scaled_target[time_steps:]

        # Model selection
        st.write("### Select Models for Prediction:")
        selected_models = st.multiselect(
            "Choose one or more models:",
            ["RNN", "LSTM", "GRU"],
            default=["LSTM"]
        )

        # Predictions
        predictions = {}
        if "RNN" in selected_models:
            predictions['RNN'] = scaler_y.inverse_transform(rnn_model.predict(X))
        if "LSTM" in selected_models:
            predictions['LSTM'] = scaler_y.inverse_transform(lstm_model.predict(X))
        if "GRU" in selected_models:
            predictions['GRU'] = scaler_y.inverse_transform(gru_model.predict(X))

        # Plot results
        st.write("### Prediction Results:")
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[time_steps:], scaler_y.inverse_transform(true_values), label="True Values", color='black', linewidth=2)
        for model_name, model_predictions in predictions.items():
            plt.plot(df.index[time_steps:], model_predictions, label=f"{model_name} Predictions")
        plt.title("Model Predictions vs True Values")
        plt.xlabel("Time")
        plt.ylabel("Potencia")
        plt.legend()
        plt.grid(True)

        # Display the plot in Streamlit
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf)

        # Option to download predictions
        st.write("### Download Predictions")
        for model_name, model_predictions in predictions.items():
            pred_df = pd.DataFrame({
                "DATETIME": df.index[time_steps:],
                "True Values": scaler_y.inverse_transform(true_values).flatten(),
                f"{model_name} Predictions": model_predictions.flatten()
            })
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download {model_name} Predictions",
                data=csv,
                file_name=f"{model_name}_predictions.csv",
                mime="text/csv"
            )

