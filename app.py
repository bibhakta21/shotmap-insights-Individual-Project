# python -m streamlit run app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Shot Selection Analytics", 
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        padding: 2rem 3rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main title styling */
    .main-title {
        text-align: center;
        color: #1e293b;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        color: white;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #10b981;
    }
    
    .subsection-header {
        color: #374151;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    .sidebar-header {
        color: #1e293b;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding: 1rem 0;
        border-bottom: 2px solid #10b981;
    }
    
    /* Metrics styling */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #10b981;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-title">⚽ Shot Selection Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced football analytics for optimal shot selection and performance optimization</p>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('clean_shots_final.csv')
    df['is_goal'] = df['shot.outcome.name'].apply(lambda x: 1 if x == 'Goal' else 0)
    df['distance_to_goal'] = np.sqrt((120 - df['x'])**2 + (40 - df['y'])**2)
    return df

df = load_data()

# Enhanced sidebar with icons and styling
st.sidebar.markdown('<h2 class="sidebar-header">🎯 Dashboard Controls</h2>', unsafe_allow_html=True)

team_list = ["All Teams"] + sorted(df['team.name'].dropna().unique().tolist())
default_index = team_list.index("England") if "England" in team_list else 0
team_filter = st.sidebar.selectbox(
    "🏆 Select Team", 
    team_list, 
    index=default_index,
    help="Filter data by specific team or view all teams"
)

# Additional filter options
outcome_filter = st.sidebar.multiselect(
    "🎯 Shot Outcomes",
    options=df['shot.outcome.name'].unique(),
    default=df['shot.outcome.name'].unique(),
    help="Filter by shot outcomes"
)

xg_range = st.sidebar.slider(
    "📊 xG Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.05,
    help="Filter shots by Expected Goals value"
)

# Apply filters
filtered_df = df.copy()
if team_filter != "All Teams":
    filtered_df = filtered_df[filtered_df['team.name'] == team_filter]
filtered_df = filtered_df[filtered_df['shot.outcome.name'].isin(outcome_filter)]
filtered_df = filtered_df[
    (filtered_df['shot.statsbomb_xg'] >= xg_range[0]) & 
    (filtered_df['shot.statsbomb_xg'] <= xg_range[1])
]

