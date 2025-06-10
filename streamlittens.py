import streamlit as st
import pickle, seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


st.set_page_config("Predicción Consumo Combustible", ":fuelpump:", layout="centered")
st.title("🚗 Predicción de Consumo de Combustible (MPG)")


@st.cache_data
def cargar_datos():
    return sns.load_dataset("mpg").dropna()
df = cargar_datos()


def entrenar_modelos(df):
    X, y = df[["weight"]], df["mpg"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    modelos = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1)
    }
    for k in modelos:
        modelos[k].fit(X_scaled, y)
    return scaler, modelos


try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
    modelos = {
        "Linear": pickle.load(open("model_lr.pkl", "rb")),
        "Ridge": pickle.load(open("model_ridge.pkl", "rb")),
        "Lasso": pickle.load(open("model_lasso.pkl", "rb"))
    }
except:
    scaler, modelos = entrenar_modelos(df)
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(modelos["Linear"], open("model_lr.pkl", "wb"))
    pickle.dump(modelos["Ridge"], open("model_ridge.pkl", "wb"))
    pickle.dump(modelos["Lasso"], open("model_lasso.pkl", "wb"))


st.sidebar.header("⚙️ Parámetros")
peso_input = st.sidebar.slider("Peso del Vehículo", float(df.weight.min()), float(df.weight.max()), float(df.weight.mean()))
if st.sidebar.button("🔄 Reentrenar modelos"):
    scaler, modelos = entrenar_modelos(df)
    for k in modelos:
        pickle.dump(modelos[k], open(f"model_{k.lower()}.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    st.sidebar.success("Modelos reentrenados")


st.subheader("📊 Predicción de MPG con Modelos Lineales")
X_scaled_input = scaler.transform([[peso_input]])
cols = st.columns(3)
for i, (nombre, modelo) in enumerate(modelos.items()):
    pred = modelo.predict(X_scaled_input)[0]
    with cols[i]:
        st.markdown(f"**{nombre}**")
        st.metric("Predicción MPG", f"{pred:.2f}")
        st.caption(f"Coef: {modelo.coef_[0]:.4f}, Intercepto: {modelo.intercept_:.2f}")


def graficar_predicciones(modelo, nombre):
    X = df[["weight"]]
    y_true = df["mpg"]
    y_pred = modelo.predict(scaler.transform(X))


    fig, ax = plt.subplots()
    ax.scatter(X, y_true, alpha=0.4, label="Datos reales")
    ax.plot(X, y_pred, color="r", label=f"{nombre}")
    ax.set(title=f"{nombre}: Predicción vs Real", xlabel="Peso", ylabel="MPG")
    ax.legend()
    st.pyplot(fig)


for nombre, modelo in modelos.items():
    with st.expander(f"📈 {nombre}"):
        graficar_predicciones(modelo, nombre)
        y_pred = modelo.predict(scaler.transform(df[["weight"]]))
        st.write(f"MSE: {mean_squared_error(df['mpg'], y_pred):.2f}")
        st.write(f"MAE: {mean_absolute_error(df['mpg'], y_pred):.2f}")
        st.write(f"R²: {r2_score(df['mpg'], y_pred):.2f}")


# Nuevos gráficos exploratorios
with st.expander("📊 Distribución de Variables"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["weight"], bins=30, ax=axs[0], color="skyblue")
    axs[0].set_title("Histograma: Peso del Vehículo")
    sns.histplot(df["mpg"], bins=30, ax=axs[1], color="lightgreen")
    axs[1].set_title("Histograma: MPG")
    st.pyplot(fig)


    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x=df["weight"], ax=axs2[0], color="skyblue")
    axs2[0].set_title("Boxplot: Peso del Vehículo")
    sns.boxplot(x=df["mpg"], ax=axs2[1], color="lightgreen")
    axs2[1].set_title("Boxplot: MPG")
    st.pyplot(fig2)


# MLP
st.sidebar.header("🧠 MLP Config")
optimizer_name = st.sidebar.selectbox("Optimizador", ["adam", "sgd", "rmsprop"])
loss_reg = st.sidebar.selectbox("Pérdida Regresión", ["mse", "mae"])
epochs = st.sidebar.slider("Épocas", 10, 200, 50, 10)
batch = st.sidebar.slider("Batch Size", 8, 64, 16, 8)


opt_map = {"adam": Adam(), "sgd": SGD(), "rmsprop": RMSprop()}
X_mlp = StandardScaler().fit_transform(df[["weight"]])
y_mlp = df["mpg"]
X_train, X_test, y_train, y_test = train_test_split(X_mlp, y_mlp, test_size=0.2, random_state=42)


if st.sidebar.button("🚀 Entrenar MLP"):
    model_mlp = Sequential([
        Dense(16, activation="relu", input_dim=1),
        Dense(8, activation="relu"),
        Dense(1)
    ])
    model_mlp.compile(optimizer=opt_map[optimizer_name], loss=loss_reg, metrics=["mae", "mse"])
    hist = model_mlp.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=0.2, verbose=0)


    st.subheader("📈 Curva de Entrenamiento MLP")
    fig, ax = plt.subplots()
    ax.plot(hist.history["loss"], label="Train Loss")
    ax.plot(hist.history["val_loss"], label="Val Loss")
    ax.plot(hist.history["mae"], label="Train MAE")
    ax.plot(hist.history["val_mae"], label="Val MAE")
    ax.set_title("MLP - Regresión")
    ax.legend()
    st.pyplot(fig)


    loss, mae, mse = model_mlp.evaluate(X_test, y_test, verbose=0)
    st.success(f"Evaluación MLP - {loss_reg.upper()}: {loss:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}")


    pred_mlp = model_mlp.predict(scaler.transform([[peso_input]]))[0][0]
    st.success(f"🔮 Predicción MLP para {peso_input:.0f} lbs: {pred_mlp:.2f} MPG")


    # Comparación visual MLP vs lineal
    y_pred_mlp = model_mlp.predict(X_mlp).flatten()
    y_pred_lr = modelos["Linear"].predict(scaler.transform(df[["weight"]]))


    fig, axs = plt.subplots(1, 2, figsize=(14,5))
    axs[0].scatter(df["weight"], df["mpg"], alpha=0.4, label="Datos")
    axs[0].plot(df["weight"], y_pred_mlp, label="MLP", color="green")
    axs[0].plot(df["weight"], y_pred_lr, label="Lineal", color="red")
    axs[0].set(title="MLP vs Regresión Lineal", xlabel="Peso", ylabel="MPG")
    axs[0].legend()


    # Gráfico residuales
    resid_mlp = df["mpg"] - y_pred_mlp
    resid_lr = df["mpg"] - y_pred_lr
    axs[1].scatter(df["weight"], resid_mlp, label="Residuales MLP", alpha=0.6)
    axs[1].scatter(df["weight"], resid_lr, label="Residuales Lineal", alpha=0.6)
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].set(title="Residuales vs Peso", xlabel="Peso", ylabel="Residuales")
    axs[1].legend()


    st.pyplot(fig)


    # Heatmap de predicción MLP
    df_heat = df.copy()
    df_heat["Predicción MLP"] = y_pred_mlp
    fig, ax = plt.subplots(figsize=(10,6))
    scatter = ax.scatter(df_heat["weight"], df_heat["mpg"], c=df_heat["Predicción MLP"], cmap="viridis", alpha=0.7)
    ax.set(title="Dispersión MPG con color según Predicción MLP", xlabel="Peso", ylabel="MPG")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Predicción MLP")
    st.pyplot(fig)
