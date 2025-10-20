import streamlit as st
import pandas as pd
import cloudpickle

@st.cache_data
def load_data(filepath):
    """
    Carga el CSV de datos crudos desde una ruta.
    Utiliza el caché de Streamlit para evitar recargar en cada interacción.
    """
    try:
        df = pd.read_csv(filepath)
        # Limpieza básica para las visualizaciones del EDA
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos en '{filepath}'.")
        st.info("Asegúrate de que 'telco_churn.csv' esté en la misma carpeta que la app.")
        return None

@st.cache_resource
def load_model(model_path):
    """
    Carga un pipeline .joblib o .pkl desde la ruta especificada.
    Utiliza el caché de recursos de Streamlit, ideal para objetos pesados como modelos.
    """
    try:
        with open(model_path, 'rb') as f:
            model = cloudpickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el modelo en '{model_path}'.")
        st.info(f"Verifica que el archivo exista y la ruta sea correcta.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el modelo en '{model_path}': {e}")
        return None
