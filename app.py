#python -m streamlit run app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans


st.set_page_config(layout="wide", page_title="Shot Selection Dashboard")
st.markdown("<h1 style='text-align: center;'>Shot Selection Optimization Dashboard</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('clean_shots_final.csv')
    df['is_goal'] = df['shot.outcome.name'].apply(lambda x: 1 if x == 'Goal' else 0)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
team_list = ["All"] + sorted(df['team.name'].dropna().unique().tolist())
default_index = team_list.index("England") if "England" in team_list else 0
team_filter = st.sidebar.selectbox("Select Team", team_list, index=default_index)

filtered_df = df if team_filter == "All" else df[df['team.name'] == team_filter]

# Pitch setup
pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')


# Heatmap and Outcome Mapping
st.subheader("Shot Location Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig, ax = pitch.draw(figsize=(7, 5))
    pitch.kdeplot(filtered_df['x'], filtered_df['y'], ax=ax, fill=True, cmap='Reds', thresh=0.05, bw_adjust=0.9)
    ax.set_title("Shot Density Heatmap", fontsize=12)
    st.pyplot(fig)

with col2:
    fig2, ax2 = pitch.draw(figsize=(7, 5))
    outcome_styles = {
        'Saved': ('blue', 0.5),
        'Blocked': ('orange', 0.5),
        'Off T': ('red', 0.15),
        'Missed': ('black', 0.2),
        'Goal': ('green', 0.85)
    }
    for outcome, (color, alpha) in outcome_styles.items():
        sub_df = filtered_df[filtered_df['shot.outcome.name'] == outcome]
        pitch.scatter(sub_df['x'], sub_df['y'], ax=ax2, color=color, alpha=alpha, s=15, label=outcome)
    ax2.set_title("Shot Outcome Distribution", fontsize=12)
    ax2.legend(loc='upper center', ncol=5)
    st.pyplot(fig2)


# xG Insights
st.subheader("Expected Goals (xG) Analysis")

col3, col4 = st.columns(2)

with col3:
    filtered_df['is_goal_str'] = filtered_df['is_goal'].astype(str)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    sns.stripplot(data=filtered_df, x='is_goal_str', y='shot.statsbomb_xg', jitter=0.25,
                  palette={'1': 'green', '0': 'red'}, ax=ax3)
    ax3.set_xticklabels(['No Goal', 'Goal'])
    ax3.set_title("xG by Goal Outcome")
    ax3.set_xlabel("Outcome")
    ax3.set_ylabel("Expected Goals")
    ax3.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig3)

with col4:
    filtered_df['distance_to_goal'] = np.sqrt((120 - filtered_df['x'])**2 + (40 - filtered_df['y'])**2)
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    sns.regplot(data=filtered_df, x='distance_to_goal', y='shot.statsbomb_xg',
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'blue'}, ax=ax4)
    ax4.set_title("Distance to Goal vs xG")
    ax4.set_xlabel("Distance to Goal")
    ax4.set_ylabel("Expected Goals")
    ax4.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig4)


# Pressure vs Outcome
st.subheader("Shot Outcome Under Pressure")

filtered_df['pressure_label'] = filtered_df['under_pressure'].apply(lambda x: "Under Pressure" if x else "Not Under Pressure")

col5, col6 = st.columns(2)

with col5:
    goal_rate = filtered_df.groupby('pressure_label')['is_goal'].mean().reset_index()
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=goal_rate, x='pressure_label', y='is_goal', palette='Set2', ax=ax5)
    ax5.set_title("Goal Rate vs Pressure")
    ax5.set_ylabel("Goal Rate")
    ax5.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig5)

with col6:
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=filtered_df, x='pressure_label', y='shot.statsbomb_xg', palette='Set2', ax=ax6)
    ax6.set_title("xG vs Pressure")
    ax6.set_ylabel("Expected Goals")
    ax6.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig6)


# KMeans Clustering
st.subheader("K-Means Shot Clustering")

kmeans = KMeans(n_clusters=5, random_state=42)
filtered_df['cluster'] = kmeans.fit_predict(filtered_df[['x', 'y']])

fig7, ax7 = pitch.draw(figsize=(8, 5))
pitch.scatter(filtered_df['x'], filtered_df['y'], c=filtered_df['cluster'], cmap='tab10', ax=ax7, edgecolors='black')
ax7.set_title("Shot Location Clusters")
st.pyplot(fig7)

st.markdown("### Cluster-wise Summary")
summary = filtered_df.groupby('cluster').agg(
    avg_xg=('shot.statsbomb_xg', 'mean'),
    goal_rate=('is_goal', 'mean'),
    shots=('x', 'count')
).reset_index().sort_values(by='goal_rate', ascending=False)
st.dataframe(summary.style.background_gradient(cmap='Blues'))
