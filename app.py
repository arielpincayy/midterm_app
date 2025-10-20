import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

# Importar las funciones desde nuestro archivo utils
from utils import load_data, load_model

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard de Churn 📞",
    page_icon="💼",
    layout="wide"
)

# --- Carga de Datos y Modelos ---
df = load_data('telco_churn.csv')
if df is None:
    st.stop()

model_files = {
    "Random Forest (Pipeline Completo)": "models/rf_pipeline.joblib",
    "Stacking (con PCA)": "models/stack_pca.joblib",
    "Stacking (Features Seleccionadas)": "models/stack_selected.joblib"
}

# --- Preparación de Datos para Métricas ---
# Creamos el split de manera determinista para evaluar los modelos
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)


# --- UI Principal ---
st.title("Dashboard de Decisión sobre Fuga de Clientes (Churn)")

# --- Navegación por Pestañas ---
tab1, tab2, tab3 = st.tabs(["🔍 **Explorador y Predicción**", "💼 **Impacto de Negocio**", "📊 **Rendimiento de Modelos**"])


# =====================================================================================
# --- PESTAÑA 1: EXPLORADOR Y PREDICCIÓN ---
# =====================================================================================
with tab1:
    st.header("Análisis Exploratorio y Predicción Individual")

    # --- Barra Lateral de Filtros ---
    st.sidebar.header("Filtros Globales 🕵️")
    st.sidebar.markdown("Estos filtros afectan a la sección de exploración.")
    selected_contract = st.sidebar.multiselect(
        "Tipo de Contrato", options=df['Contract'].unique(), default=df['Contract'].unique()
    )
    selected_payment = st.sidebar.multiselect(
        "Método de Pago", options=df['PaymentMethod'].unique(), default=df['PaymentMethod'].unique()
    )
    df_filtered = df.query(
        "Contract == @selected_contract and PaymentMethod == @selected_payment"
    )

    if df_filtered.empty:
        st.warning("No hay datos que coincidan con los filtros seleccionados.")
        st.stop()

    # --- KPIs Estáticos ---
    st.subheader("Métricas Clave del Segmento")
    kpi1, kpi2 = st.columns(2)
    churn_rate_yes = df_filtered['Churn'].value_counts(normalize=True).get('Yes', 0)
    kpi1.metric("Tasa de Fuga (Churn)", f"{churn_rate_yes:.2%}")
    avg_monthly_charge = df_filtered['MonthlyCharges'].mean()
    kpi2.metric("Cargo Mensual Promedio", f"${avg_monthly_charge:.2f}")

    # --- Resultados de la sección EDA (Visualizaciones Específicas) ---
    st.subheader("Visualizaciones Específicas")
    row1_col1, row1_col2 = st.columns(2)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols.remove('customerID')
    
    with row1_col1:
        st.markdown("##### 📈 Distribución de una Variable")
        dist_var = st.selectbox("Variable:", options=numeric_cols + categorical_cols, key="dist_var")
        if dist_var in numeric_cols:
            fig = px.histogram(df_filtered, x=dist_var, color="Churn", marginal="box", color_discrete_map={'Yes': '#FF5B5B', 'No': '#00B0F0'})
        else:
            fig = px.bar(df_filtered[dist_var].value_counts().reset_index(), x=dist_var, y='count')
        st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        st.markdown("##### 📦 Box Plots (Numérico vs. Categórico)")
        cat_box = st.selectbox("Variable Categórica:", options=categorical_cols, key="cat_box", index=categorical_cols.index('Contract'))
        num_box = st.selectbox("Variable Numérica:", options=numeric_cols, key="num_box", index=numeric_cols.index('MonthlyCharges'))
        fig_box = px.box(df_filtered, x=cat_box, y=num_box, color="Churn", color_discrete_map={'Yes': '#FF5B5B', 'No': '#00B0F0'})
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    # --- Sección de Clasificación (Predicción) ---
    st.header("Predicción de Churn para un Nuevo Cliente 🔮")
    model_name_pred = st.selectbox("Elige el pipeline de predicción:", options=list(model_files.keys()), key="model_pred")
    pipeline = load_model(model_files[model_name_pred])
    
    with st.form("prediction_form"):
        input_data = {}
        form_cols = st.columns(3)
        features = df.drop(columns=['customerID', 'Churn']).columns
        for i, col in enumerate(features):
            with form_cols[i % 3]:
                if df[col].dtype == "object":
                    input_data[col] = st.selectbox(label=col, options=df[col].unique().tolist(), key=f"form_{col}")
                elif col == 'SeniorCitizen':
                    input_data[col] = st.selectbox(label=col, options=[0, 1], key=f"form_{col}")
                else:
                    input_data[col] = st.number_input(label=col, value=float(df[col].median()), key=f"form_{col}")
        submit_button = st.form_submit_button(label="Predecir Churn")

    if submit_button and pipeline:
        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)
        churn_status = "Sí (Fuga)" if prediction[0] == 1 else "No (Permanece)"
        prob_churn = prediction_proba[0][1]

        st.subheader("Resultado de la Predicción:")
        if churn_status == "Sí (Fuga)":
            st.error(f"El cliente probablemente hará Churn: **{churn_status}**")
        else:
            st.success(f"El cliente probablemente se quedará: **{churn_status}**")
        st.metric(f"Probabilidad de Fuga (Modelo: {model_name_pred})", f"{prob_churn:.2%}")
        
        # --- Automatic Recommendation ---
        if prob_churn > 0.70:
            st.warning("🚨 **ALTO RIESGO DE FUGA DETECTADO**")
            st.info("Acción sugerida: Contactar al cliente proactivamente. Ofrecer un descuento temporal, un nuevo servicio de valor agregado o una mejora en su plan actual para aumentar la retención.")

# =====================================================================================
# --- PESTAÑA 2: IMPACTO DE NEGOCIO ---
# =====================================================================================
with tab2:
    st.header("Simulación del Impacto en el Negocio")
    st.markdown("Analiza el riesgo financiero y el potencial de ahorro basado en las predicciones del modelo.")

    model_name_biz = st.selectbox("Selecciona un modelo para la simulación:", options=list(model_files.keys()), key="model_biz")
    pipeline_biz = load_model(model_files[model_name_biz])

    if pipeline_biz:
        # Predecir sobre todo el dataset
        df_biz = df.copy()
        df_biz['prediction'] = pipeline_biz.predict(df_biz)
        
        customers_at_risk = df_biz[df_biz['prediction'] == 1]
        
        # --- Business KPIs ---
        st.subheader("KPIs de Riesgo y Oportunidad")
        kpi_biz1, kpi_biz2 = st.columns(2)
        
        pct_at_risk = len(customers_at_risk) / len(df_biz)
        kpi_biz1.metric("% de Clientes en Riesgo de Fuga", f"{pct_at_risk:.2%}")

        revenue_at_risk = customers_at_risk['MonthlyCharges'].sum()
        kpi_biz2.metric("Ingresos Mensuales en Riesgo", f"${revenue_at_risk:,.2f}")
        
        st.markdown("### Simulación de Ahorro por Retención")
        retention_pct = st.slider("Selecciona el % de clientes en riesgo que podrías retener:", 1, 100, 10)
        potential_savings = revenue_at_risk * (retention_pct / 100.0)
        st.success(f"Si retienes al **{retention_pct}%** de los clientes en riesgo, podrías salvar **${potential_savings:,.2f}** en ingresos mensuales.")


# =====================================================================================
# --- PESTAÑA 3: RENDIMIENTO DE MODELOS ---
# =====================================================================================
with tab3:
    st.header("Comparación y Análisis de Rendimiento de Modelos")
    
    # --- Metrics Comparison ---
    st.subheader("📊 Comparación de Métricas (sobre datos de Test)")
    
    metrics_data = []
    for model_name, model_path in model_files.items():
        model = load_model(model_path)
        if model:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics_data.append({
                "Modelo": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_proba)
            })
    metrics_df = pd.DataFrame(metrics_data).set_index("Modelo")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"))

    st.divider()

    col_cm, col_fi = st.columns(2)

    with col_cm:
        # --- Confusion Matrix ---
        st.subheader("🔥 Matriz de Confusión")
        model_name_cm = st.selectbox("Selecciona un modelo para ver su matriz de confusión:", options=list(model_files.keys()), key="model_cm")
        model_cm = load_model(model_files[model_name_cm])
        if model_cm:
            y_pred_cm = model_cm.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_cm)
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            st.pyplot(fig_cm)

    with col_fi:
        # --- Feature Importance Plot ---
        st.subheader("⭐ Importancia de Features")
        try:
            fi_df = pd.read_csv('feature_importance_combined.csv')
        except Exception as e:
            st.warning(f"No se pudo abrir 'feature_importance_combined.csv': {e}")
        else:
            # Normalizar nombres de columnas comunes
            cols_lower = [c.lower() for c in fi_df.columns]
            if 'feature' in cols_lower and 'importance' in cols_lower:
                feat_col = fi_df.columns[cols_lower.index('feature')]
                imp_col = fi_df.columns[cols_lower.index('importance')]
            else:
                # Fallback: usar las dos primeras columnas asumidas como feature/importance
                feat_col, imp_col = fi_df.columns[0], fi_df.columns[1]

            # Asegurar que la importancia sea numérica y limpiar
            fi_df[imp_col] = pd.to_numeric(fi_df[imp_col], errors='coerce')
            fi_df = fi_df.dropna(subset=[imp_col, feat_col])

            if fi_df.empty:
                st.warning("El archivo no contiene datos válidos de importancia de features.")
            else:
                # Configurable: seleccionar top N
                max_n = len(fi_df)
                top_n = st.slider("Mostrar top N features", min_value=1, max_value=max_n, value=min(20, max_n), key="top_n_features")
                fi_sorted = fi_df.sort_values(by=imp_col, ascending=False).head(top_n)

                st.subheader("Importancia de Features")
                st.dataframe(fi_sorted.reset_index(drop=True))

                fig_fi = px.bar(
                    fi_sorted,
                    x=imp_col,
                    y=feat_col,
                    orientation='h',
                    title=f"Top {top_n} Features por Importancia",
                    color=imp_col,
                    color_continuous_scale='blues'
                )
                fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
                st.plotly_chart(fig_fi, use_container_width=True)
