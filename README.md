# Fiber-Reinforced Asphalt Concrete Multi-Target Prediction System 🚧

This is a **visual interactive web application** built with **Streamlit**, designed for predicting three key performance
indicators of **fiber-reinforced asphalt concrete**:

- **Marshall Quotient (MQ)** – A mechanical stability index
- **Marshall Stability (MS)** – Shear resistance
- **Flow Value (FV)** – Deformation capacity

The system allows users to input material properties of asphalt mixtures and provides real-time predictions for the
above three performance metrics. It also includes **SHAP explanations** and **LCA (Life Cycle Assessment)** analysis.

---

## 📌 Key Features

- 🔍 **Interactive Input Interface**: Supports input of various material parameters (asphalt, fiber, aggregates, etc.)
- 📊 **Multi-Target Joint Prediction**: Predicts MQ, MS, and FV using a Voting Regressor model
- 🧠 **SHAP Interpretation**: Visualizes feature importance and impact on model predictions
- 🌱 **LCA Analysis**: Calculates material cost, carbon footprint, and energy consumption
- 📈 **Contribution Pie Charts**: Shows the contribution of each material component to cost, carbon footprint, and energy
  consumption

---

## 🧰 Technologies Used

- **Streamlit** – For building the interactive web UI
- **Scikit-learn / VotingRegressor** – For multi-model ensemble prediction
- **SHAP** – For model interpretation
- **Dill** – For saving and loading SHAP explainers
- **Matplotlib / Pandas / NumPy** – For data processing and visualization
- **Joblib** – For model serialization

---

## 📦 Project Structure

