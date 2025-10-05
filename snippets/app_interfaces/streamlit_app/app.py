import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Интерактивный анализ датасета")

# --- Загрузка файла ---
uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")

if uploaded_file is not None:
    # читаем датасет
    df = pd.read_csv(uploaded_file)

    st.subheader("Первые строки таблицы")
    st.write(df.head())

    st.subheader("Статистика по данным")
    st.write(df.describe())

    # --- Выбор колонок для графиков ---
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if numeric_columns:
        st.subheader("Визуализация")

        x_axis = st.selectbox("Выберите X", numeric_columns)
        y_axis = st.selectbox("Выберите Y", numeric_columns)

        # --- Построение графика ---
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis], alpha=0.6)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"График: {y_axis} от {x_axis}")
        st.pyplot(fig)
    else:
        st.warning("В датасете нет числовых колонок для построения графиков.")
else:
    st.info("Загрузите CSV-файл для начала работы.")
