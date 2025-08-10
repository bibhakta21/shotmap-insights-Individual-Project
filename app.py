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
