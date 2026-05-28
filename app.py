import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Intelligent Operations Commander", layout="wide", initial_sidebar_state="expanded")

# --- CSS INJECTION ---
def inject_css():
    st.markdown("""
    <style>
    /* Dark Mode Theme & Glassmorphism */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Glowing Glassmorphism Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.5);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3), 0 0 15px rgba(88, 166, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4), 0 0 25px rgba(88, 166, 255, 0.4);
    }

    /* Anomaly Alert Glow */
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 10px rgba(255, 123, 114, 0.4); }
        50% { box-shadow: 0 0 30px rgba(255, 123, 114, 0.8), inset 0 0 10px rgba(255, 123, 114, 0.3); }
        100% { box-shadow: 0 0 10px rgba(255, 123, 114, 0.4); }
    }
    .alert-glow {
        background: rgba(40, 10, 10, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 123, 114, 0.5);
        border-radius: 10px;
        padding: 20px;
        animation: pulseGlow 2s infinite;
    }
    
    /* 3D Radar Animation */
    @keyframes scan {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .radar {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        border: 2px solid #58a6ff;
        position: relative;
        overflow: hidden;
        margin: 0 auto 20px auto;
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.5);
    }
    .radar::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        width: 50%;
        height: 50%;
        background: linear-gradient(45deg, rgba(88, 166, 255, 0.8) 0%, transparent 100%);
        transform-origin: bottom left;
        animation: scan 2s linear infinite;
    }
    .radar::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 4px;
        height: 4px;
        background: #fff;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 0 10px #fff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA GENERATION & CACHING ---
@st.cache_data
def generate_mock_data(n=1500):
    np.random.seed(42)
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    
    data = {
        'Order_ID': [f"ORD-{i:06d}" for i in range(1, n+1)],
        'Origin_City': np.random.choice(cities, n),
        'Destination_City': np.random.choice(cities, n),
        'Distance_KM': np.random.uniform(50, 3000, n),
        'Traffic_Density_Index': np.random.uniform(0.1, 1.0, n),
        'Weather_Impact_Score': np.random.uniform(0.1, 1.0, n),
        'Driver_Experience_Yrs': np.random.uniform(0.5, 30.0, n)
    }
    df = pd.DataFrame(data)
    
    # Introduce some logical correlation for the target variable and anomalies
    base_time = df['Distance_KM'] / 80.0  # avg 80 km/h
    traffic_delay = df['Traffic_Density_Index'] * 5.0
    weather_delay = df['Weather_Impact_Score'] * 8.0
    exp_reduction = df['Driver_Experience_Yrs'] * 0.1
    
    df['Actual_Delivery_Time_Hrs'] = base_time + traffic_delay + weather_delay - exp_reduction + np.random.normal(0, 2, n)
    df['Actual_Delivery_Time_Hrs'] = df['Actual_Delivery_Time_Hrs'].clip(lower=1.0)
    
    # Inject some anomalies (e.g., extreme weather or traffic despite short distance)
    anomaly_indices = np.random.choice(n, int(n * 0.05), replace=False)
    df.loc[anomaly_indices, 'Actual_Delivery_Time_Hrs'] *= np.random.uniform(2.0, 4.0, len(anomaly_indices))
    df.loc[anomaly_indices, 'Traffic_Density_Index'] = np.random.uniform(0.8, 1.0, len(anomaly_indices))
    
    return df

# --- MODELS CACHING ---
@st.cache_resource
def train_anomaly_model(df):
    features = ['Distance_KM', 'Traffic_Density_Index', 'Weather_Impact_Score', 'Actual_Delivery_Time_Hrs']
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[features])
    return model

@st.cache_resource
def train_predictive_model(df):
    features = ['Distance_KM', 'Traffic_Density_Index', 'Weather_Impact_Score', 'Driver_Experience_Yrs']
    X = df[features]
    y = df['Actual_Delivery_Time_Hrs']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- MAIN APP ---
def main():
    inject_css()
    
    st.markdown('<div class="radar"></div>', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #58a6ff;'>The Intelligent Operations Commander</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e;'>Automated, No-LLM Predictive Analytics & Insights Engine</p>", unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.markdown("### 🎛️ Command Controls")
    view = st.sidebar.radio("Select Operational View:", 
                            ["1. Data Ingestion & Health", 
                             "2. Anomaly Detection", 
                             "3. 3D Simulation Sandbox"])
    
    # File Uploader
    uploaded_file = st.sidebar.file_uploader("Upload Logistics Data (CSV)", type=['csv'], help="Upload operational records. If empty, mock data is generated.")
    
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = generate_mock_data()
            
        # Ensure required columns
        req_cols = ['Order_ID', 'Distance_KM', 'Traffic_Density_Index', 'Weather_Impact_Score', 'Driver_Experience_Yrs', 'Actual_Delivery_Time_Hrs']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return
            
        if view == "1. Data Ingestion & Health":
            st.markdown("### 📡 Smart Ingestion & Cleaning Pipeline")
            
            # Health Score Calculation
            completeness = df.notnull().mean().mean() * 100
            health_score = min(100, completeness - (df.duplicated().sum() / len(df) * 100))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="glass-card">
                    <h4>Total Records</h4>
                    <h2>{len(df):,}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="glass-card">
                    <h4>Data Health Score</h4>
                    <h2 style="color: {'#3fb950' if health_score > 90 else '#d29922'};">{health_score:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="glass-card">
                    <h4>Avg Delivery Time</h4>
                    <h2>{df['Actual_Delivery_Time_Hrs'].mean():.1f} hrs</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Raw Operational Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
        elif view == "2. Anomaly Detection":
            st.markdown("### 🚨 Unsupervised Anomaly Detection")
            
            iso_model = train_anomaly_model(df)
            features = ['Distance_KM', 'Traffic_Density_Index', 'Weather_Impact_Score', 'Actual_Delivery_Time_Hrs']
            
            df_anom = df.copy()
            df_anom['Anomaly'] = iso_model.predict(df_anom[features])
            df_anom['Anomaly_Score'] = iso_model.decision_function(df_anom[features])
            
            anomalies = df_anom[df_anom['Anomaly'] == -1]
            normal = df_anom[df_anom['Anomaly'] == 1]
            
            st.markdown(f"""
            <div class="alert-glow">
                <h3 style="margin-top: 0; color: #ff7b72;">⚠️ Critical Bottlenecks Detected</h3>
                <p>Isolated <b>{len(anomalies)}</b> operational anomalies requiring immediate routing intervention.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("<br>", unsafe_allow_html=True)
            
            # 3D Depth Scatter via Plotly
            fig = px.scatter(df_anom, x='Distance_KM', y='Actual_Delivery_Time_Hrs', 
                             color='Anomaly', color_discrete_map={1: '#58a6ff', -1: '#ff7b72'},
                             size='Traffic_Density_Index', hover_data=['Order_ID', 'Weather_Impact_Score'],
                             title="Operational Matrix: Distance vs Delivery Time",
                             template="plotly_dark")
            
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Flagged Shipments")
            st.dataframe(anomalies.sort_values('Anomaly_Score').head(50), use_container_width=True)
            
        elif view == "3. 3D Simulation Sandbox":
            st.markdown("### 🌌 Predictive Modeling & 3D Simulation Sandbox")
            
            model = train_predictive_model(df)
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("#### 🔧 Stress Test Parameters")
            surge = st.sidebar.slider("Global Traffic Surge (%)", -50, 100, 0, help="Modify traffic density macro coefficient.")
            weather_delay = st.sidebar.slider("Severe Weather Factor (%)", -50, 100, 0, help="Modify weather impact macro coefficient.")
            driver_shortage = st.sidebar.slider("Driver Shortage Penalty (%)", 0, 100, 0, help="Reduces effective driver experience.")
            
            df_sim = df.copy()
            df_sim['Traffic_Density_Index'] = df_sim['Traffic_Density_Index'] * (1 + surge/100.0)
            df_sim['Weather_Impact_Score'] = df_sim['Weather_Impact_Score'] * (1 + weather_delay/100.0)
            df_sim['Driver_Experience_Yrs'] = df_sim['Driver_Experience_Yrs'] * (1 - driver_shortage/100.0)
            
            # Cap limits
            df_sim['Traffic_Density_Index'] = df_sim['Traffic_Density_Index'].clip(0, 1)
            df_sim['Weather_Impact_Score'] = df_sim['Weather_Impact_Score'].clip(0, 1)
            df_sim['Driver_Experience_Yrs'] = df_sim['Driver_Experience_Yrs'].clip(lower=0.1)
            
            features = ['Distance_KM', 'Traffic_Density_Index', 'Weather_Impact_Score', 'Driver_Experience_Yrs']
            
            # Predict
            df_sim['Simulated_Delivery_Hrs'] = model.predict(df_sim[features])
            
            col1, col2 = st.columns(2)
            avg_base = df['Actual_Delivery_Time_Hrs'].mean()
            avg_sim = df_sim['Simulated_Delivery_Hrs'].mean()
            delta = avg_sim - avg_base
            
            with col1:
                st.markdown(f"""
                <div class="glass-card">
                    <h4>Baseline Avg Time</h4>
                    <h2>{avg_base:.1f} hrs</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="glass-card">
                    <h4>Simulated Avg Time</h4>
                    <h2 style="color: {'#ff7b72' if delta > 0 else '#3fb950'};">{avg_sim:.1f} hrs 
                    <span style="font-size: 16px;">({'+' if delta > 0 else ''}{delta:.1f} hrs)</span></h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("<br>", unsafe_allow_html=True)
            
            # 3D Surface / Scatter Plot
            st.markdown("#### 🧊 3D Operational Risk Landscape")
            
            # Subsample for performance in 3D
            sample = df_sim.sample(n=min(500, len(df_sim)), random_state=42)
            
            fig3d = go.Figure(data=[go.Scatter3d(
                x=sample['Distance_KM'],
                y=sample['Traffic_Density_Index'],
                z=sample['Simulated_Delivery_Hrs'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=sample['Simulated_Delivery_Hrs'],
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=sample['Order_ID']
            )])
            
            fig3d.update_layout(
                scene=dict(
                    xaxis_title='Distance (KM)',
                    yaxis_title='Traffic Density',
                    zaxis_title='Predicted Delay (Hrs)',
                    bgcolor='rgba(0,0,0,0)'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor='rgba(0,0,0,0)',
                template='plotly_dark'
            )
            st.plotly_chart(fig3d, use_container_width=True)

            # Line Chart Comparison
            st.markdown("#### 📈 Baseline vs Simulated Stress (Top 100 Routes)")
            line_df = pd.DataFrame({
                'Route_Index': range(100),
                'Baseline': df['Actual_Delivery_Time_Hrs'].head(100).values,
                'Simulated': df_sim['Simulated_Delivery_Hrs'].head(100).values
            })
            
            fig_line = px.line(line_df, x='Route_Index', y=['Baseline', 'Simulated'],
                               labels={'value': 'Delivery Time (Hrs)', 'variable': 'Scenario'},
                               template='plotly_dark')
            fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_line, use_container_width=True)
            
    except Exception as e:
        st.error("Engine Encountered a Critical Fault. Resetting Matrix...")
        st.write(f"<!-- {str(e)} -->", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #8b949e; font-size: 12px;'>SYSTEM: SECURE // DEPLOYMENT: LOCAL TARGET // PROPERTY OF INTELLIGENT OPERATIONS COMMANDER v1.0</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
