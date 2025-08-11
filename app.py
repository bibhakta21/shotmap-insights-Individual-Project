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

st.set_page_config(
    layout="wide", 
    page_title="Shot Selection Analytics", 
    initial_sidebar_state="expanded"
)

# Custom CSS
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


st.markdown('<h1 class="main-title">⚽ Shot Selection Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced football analytics for optimal shot selection and performance optimization</p>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        
        df = pd.read_csv('https://media.githubusercontent.com/media/bibhakta21/shotmap-insights-Individual-Project/refs/heads/main/clean_shots_final_streamlit.csv')

        
        st.write(" CSV Columns Found:", list(df.columns))
        st.write(f"CSV Shape: {df.shape}")

        
        if 'shot.outcome.name' not in df.columns:
            st.error("'shot.outcome.name' column not found in CSV. Please check your file.")
            st.stop()

        # Create derived columns
        df['is_goal'] = df['shot.outcome.name'].apply(lambda x: 1 if x == 'Goal' else 0)
        df['distance_to_goal'] = np.sqrt((120 - df['x'])**2 + (40 - df['y'])**2)

        return df

    except FileNotFoundError:
        st.error("CSV file not found! Ensure 'clean_shots_final.csv' is in the same directory as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the CSV: {e}")
        st.stop()

df = load_data()

#sidebar 
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

# Apply filters logic
filtered_df = df.copy()
if team_filter != "All Teams":
    filtered_df = filtered_df[filtered_df['team.name'] == team_filter]
filtered_df = filtered_df[filtered_df['shot.outcome.name'].isin(outcome_filter)]
filtered_df = filtered_df[
    (filtered_df['shot.statsbomb_xg'] >= xg_range[0]) & 
    (filtered_df['shot.statsbomb_xg'] <= xg_range[1])
]

# Shot Location Visualizations
st.markdown('<div class="section-header">🗺️ Shot Location Analysis</div>', unsafe_allow_html=True)

pitch_col1, pitch_col2 = st.columns(2)

with pitch_col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">Shot Density Heatmap</h3>', unsafe_allow_html=True)
    
    # Create matplotlib figure with custom styling
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#f8fafc', line_color='#334155', linewidth=2)
    fig, ax = pitch.draw(figsize=(8, 6))
    pitch.kdeplot(filtered_df['x'], filtered_df['y'], ax=ax, fill=True, 
                  cmap='RdYlGn_r', thresh=0.05, bw_adjust=0.9, alpha=0.8)
    ax.set_title("", fontsize=0)  # Remove default title
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pitch_col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">Shot Outcome Distribution</h3>', unsafe_allow_html=True)
    
    pitch2 = Pitch(pitch_type='statsbomb', pitch_color='#f8fafc', line_color='#334155', linewidth=2)
    fig2, ax2 = pitch2.draw(figsize=(8, 6))
    
    # Enhanced outcome colors
    outcome_styles = {
        'Saved': ('#3b82f6', 0.6),      
        'Blocked': ('#f59e0b', 0.6),      
        'Off T': ('#ef4444', 0.4),       
        'Missed': ('#6b7280', 0.4),      
        'Goal': ('#10b981', 0.9)         
    }
    
    for outcome, (color, alpha) in outcome_styles.items():
        sub_df = filtered_df[filtered_df['shot.outcome.name'] == outcome]
        if len(sub_df) > 0:
            pitch2.scatter(sub_df['x'], sub_df['y'], ax=ax2, color=color, 
                          alpha=alpha, s=20, label=outcome, edgecolors='white', linewidth=0.5)
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, 
               frameon=False, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Interactive xG Analysis
st.markdown('<div class="section-header">📊 Expected Goals (xG) Interactive Analysis</div>', unsafe_allow_html=True)

xg_col1, xg_col2 = st.columns(2)

with xg_col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">xG Distribution by Outcome</h3>', unsafe_allow_html=True)
    
    # Interactive box plot with Plotly
    fig_box = go.Figure()
    
    for outcome in filtered_df['shot.outcome.name'].unique():
        outcome_data = filtered_df[filtered_df['shot.outcome.name'] == outcome]
        color = {'Goal': '#10b981', 'Saved': '#3b82f6', 'Blocked': '#f59e0b', 
                'Off T': '#ef4444', 'Missed': '#6b7280'}.get(outcome, '#64748b')
        
        fig_box.add_trace(go.Box(
            y=outcome_data['shot.statsbomb_xg'],
            name=outcome,
            marker_color=color,
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig_box.update_layout(
        title="",
        yaxis_title="Expected Goals (xG)",
        xaxis_title="Shot Outcome",
        template="plotly_white",
        height=400,
        showlegend=False,
        font=dict(family="Inter", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with xg_col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">Distance vs xG Relationship</h3>', unsafe_allow_html=True)
    
    # Interactive scatter plot
    fig_scatter = px.scatter(
        filtered_df, 
        x='distance_to_goal', 
        y='shot.statsbomb_xg',
        color='is_goal',
        color_discrete_map={0: '#ef4444', 1: '#10b981'},
        hover_data={
            'shot.outcome.name': True,
            'distance_to_goal': ':.1f',
            'shot.statsbomb_xg': ':.3f'
        },
        labels={
            'distance_to_goal': 'Distance to Goal (yards)',
            'shot.statsbomb_xg': 'Expected Goals (xG)',
            'is_goal': 'Goal Scored'
        },
        trendline="ols",
        trendline_color_override="#64748b"
    )
    
    fig_scatter.update_layout(
        title="",
        template="plotly_white",
        height=400,
        font=dict(family="Inter", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=""
        )
    )
    
    fig_scatter.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Pressure Analysis
st.markdown('<div class="section-header">💪 Performance Under Pressure</div>', unsafe_allow_html=True)

pressure_col1, pressure_col2 = st.columns(2)

filtered_df['pressure_label'] = filtered_df['under_pressure'].apply(
    lambda x: "Under Pressure" if x else "Not Under Pressure"
)

with pressure_col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">Goal Conversion Rate</h3>', unsafe_allow_html=True)
    
    goal_rate = filtered_df.groupby('pressure_label')['is_goal'].agg(['mean', 'count']).reset_index()
    goal_rate.columns = ['pressure_label', 'goal_rate', 'shot_count']
    
    fig_bar = px.bar(
        goal_rate, 
        x='pressure_label', 
        y='goal_rate',
        color='goal_rate',
        color_continuous_scale='RdYlGn',
        text=[f'{rate:.1%}<br>({count} shots)' for rate, count in zip(goal_rate['goal_rate'], goal_rate['shot_count'])],
        labels={
            'pressure_label': 'Pressure Situation',
            'goal_rate': 'Goal Conversion Rate'
        }
    )
    
    fig_bar.update_layout(
        title="",
        template="plotly_white",
        height=400,
        showlegend=False,
        font=dict(family="Inter", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat='.1%')
    )
    
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pressure_col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">xG Quality Under Pressure</h3>', unsafe_allow_html=True)
    
    fig_violin = go.Figure()
    
    for pressure in filtered_df['pressure_label'].unique():
        pressure_data = filtered_df[filtered_df['pressure_label'] == pressure]
        color = '#ef4444' if pressure == 'Under Pressure' else '#10b981'
        
        fig_violin.add_trace(go.Violin(
            y=pressure_data['shot.statsbomb_xg'],
            name=pressure,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.7,
            line_color=color
        ))
    
    fig_violin.update_layout(
        title="",
        yaxis_title="Expected Goals (xG)",
        template="plotly_white",
        height=400,
        showlegend=False,
        font=dict(family="Inter", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_violin, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Advanced Clustering Analysis
st.markdown('<div class="section-header">🎯 Shot Zone Clustering Analysis</div>', unsafe_allow_html=True)

cluster_col1, cluster_col2 = st.columns([3, 2])

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
filtered_df['cluster'] = kmeans.fit_predict(filtered_df[['x', 'y']])

with cluster_col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">Shot Location Clusters</h3>', unsafe_allow_html=True)
    
    # Enhanced pitch visualization
    pitch3 = Pitch(pitch_type='statsbomb', pitch_color='#f8fafc', line_color='#334155', linewidth=2)
    fig3, ax3 = pitch3.draw(figsize=(10, 6))
    
    # Use a more sophisticated color palette
    colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6']
    
    for cluster in sorted(filtered_df['cluster'].unique()):
        cluster_data = filtered_df[filtered_df['cluster'] == cluster]
        pitch3.scatter(cluster_data['x'], cluster_data['y'], 
                      color=colors[cluster % len(colors)], 
                      ax=ax3, s=25, alpha=0.7, 
                      edgecolors='white', linewidth=0.5,
                      label=f'Zone {cluster + 1}')
    
    # Add cluster centroids
    centroids = kmeans.cluster_centers_
    pitch3.scatter(centroids[:, 0], centroids[:, 1], 
                  color='black', marker='X', s=200, 
                  edgecolors='white', linewidth=2, ax=ax3,
                  label='Zone Centers')
    
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=6, frameon=False, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with cluster_col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">Zone Performance Summary</h3>', unsafe_allow_html=True)
    
    # Enhanced cluster summary
    summary = filtered_df.groupby('cluster').agg(
        avg_xg=('shot.statsbomb_xg', 'mean'),
        goal_rate=('is_goal', 'mean'),
        shots=('x', 'count'),
        avg_distance=('distance_to_goal', 'mean')
    ).reset_index()
    
    summary['zone'] = 'Zone ' + (summary['cluster'] + 1).astype(str)
    summary = summary.sort_values(by='goal_rate', ascending=False)
    
    # Format for display
    display_summary = summary[['zone', 'shots', 'goal_rate', 'avg_xg', 'avg_distance']].copy()
    display_summary['goal_rate'] = (display_summary['goal_rate'] * 100).round(1)
    display_summary['avg_xg'] = display_summary['avg_xg'].round(3)
    display_summary['avg_distance'] = display_summary['avg_distance'].round(1)
    
    display_summary.columns = ['Zone', 'Shots', 'Goal Rate (%)', 'Avg xG', 'Avg Distance']
    
    st.dataframe(
        display_summary.style.background_gradient(
            subset=['Goal Rate (%)'], cmap='RdYlGn', vmin=0, vmax=100
        ).background_gradient(
            subset=['Avg xG'], cmap='Blues', vmin=0, vmax=1
        ).format({
            'Goal Rate (%)': '{:.1f}%',
            'Avg xG': '{:.3f}',
            'Avg Distance': '{:.1f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Interactive zone performance chart
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.markdown('<h3 class="subsection-header">Zone Efficiency Matrix</h3>', unsafe_allow_html=True)

fig_bubble = px.scatter(
    summary,
    x='avg_xg',
    y='goal_rate',
    size='shots',
    color='zone',
    hover_data={
        'avg_distance': ':.1f',
        'shots': True,
        'goal_rate': ':.1%',
        'avg_xg': ':.3f'
    },
    labels={
        'avg_xg': 'Average xG per Shot',
        'goal_rate': 'Goal Conversion Rate',
        'zone': 'Shot Zone'
    },
    color_discrete_sequence=colors
)

fig_bubble.update_layout(
    title="",
    template="plotly_white",
    height=500,
    font=dict(family="Inter", size=12),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    yaxis=dict(tickformat='.1%'),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig_bubble.update_traces(
    marker=dict(
        line=dict(width=2, color='white'),
        sizemode='diameter',
        sizeref=2.*max(summary['shots'])/(40.**2),
        sizemin=4
    )
)

st.plotly_chart(fig_bubble, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Analytics Section
st.markdown('<div class="section-header">🔬 Advanced Shot Analytics</div>', unsafe_allow_html=True)

advanced_col1, advanced_col2 = st.columns(2)

with advanced_col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">Shot Quality vs Volume by Team</h3>', unsafe_allow_html=True)
    
    if team_filter == "All Teams":
        team_summary = df.groupby('team.name').agg(
            avg_xg=('shot.statsbomb_xg', 'mean'),
            total_shots=('x', 'count'),
            goals=('is_goal', 'sum')
        ).reset_index()
        team_summary['goal_rate'] = team_summary['goals'] / team_summary['total_shots']
        
        fig_teams = px.scatter(
            team_summary.head(20),  # Top 20 teams
            x='total_shots',
            y='avg_xg',
            size='goal_rate',
            color='goal_rate',
            hover_name='team.name',
            color_continuous_scale='RdYlGn',
            labels={
                'total_shots': 'Total Shots',
                'avg_xg': 'Average xG per Shot',
                'goal_rate': 'Conversion Rate'
            }
        )
        
        fig_teams.update_layout(
            title="",
            template="plotly_white",
            height=400,
            font=dict(family="Inter", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_teams, use_container_width=True)
    else:
        st.info("Select 'All Teams' to view team comparison analysis")
    
    st.markdown('</div>', unsafe_allow_html=True)

with advanced_col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">Shot Angle Analysis</h3>', unsafe_allow_html=True)
    
    # Calculate shot angle
    filtered_df['shot_angle'] = np.degrees(np.arctan2(
        abs(filtered_df['y'] - 40), 
        120 - filtered_df['x']
    ))
    
    fig_angle = px.histogram(
        filtered_df,
        x='shot_angle',
        color='shot.outcome.name',
        color_discrete_map={
            'Goal': '#10b981',
            'Saved': '#3b82f6', 
            'Blocked': '#f59e0b',
            'Off T': '#ef4444',
            'Missed': '#6b7280'
        },
        labels={
            'shot_angle': 'Shot Angle (degrees)',
            'count': 'Number of Shots'
        },
        opacity=0.8
    )
    
    fig_angle.update_layout(
        title="",
        template="plotly_white",
        height=400,
        font=dict(family="Inter", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(title="Outcome", title_font_size=12)
    )
    
    st.plotly_chart(fig_angle, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with insights
st.markdown("---")
st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)

insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    best_zone = summary.loc[summary['goal_rate'].idxmax(), 'zone']
    best_rate = summary['goal_rate'].max()
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #10b981; font-size: 1.1rem; font-weight: 600;">🎯 Most Efficient Zone</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0.5rem 0;">{best_zone}</div>
        <div style="color: #64748b;">{best_rate:.1%} conversion rate</div>
    </div>
    """, unsafe_allow_html=True)

with insight_col2:
    total_shots = len(filtered_df)
    high_xg_shots = len(filtered_df[filtered_df['shot.statsbomb_xg'] > 0.3])
    high_xg_rate = (high_xg_shots / total_shots * 100) if total_shots > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #3b82f6; font-size: 1.1rem; font-weight: 600;">⭐ High Quality Shots</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0.5rem 0;">{high_xg_rate:.1f}%</div>
        <div style="color: #64748b;">Shots with xG > 0.3</div>
    </div>
    """, unsafe_allow_html=True)

with insight_col3:
    pressure_impact = filtered_df.groupby('pressure_label')['shot.statsbomb_xg'].mean()
    if len(pressure_impact) == 2:
        impact = ((pressure_impact['Not Under Pressure'] - pressure_impact['Under Pressure']) * 100)
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #f59e0b; font-size: 1.1rem; font-weight: 600;">⚡ Pressure Impact</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0.5rem 0;">{impact:+.1f}%</div>
            <div style="color: #64748b;">xG difference without pressure</div>
        </div>
        """, unsafe_allow_html=True)

# Data export option
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📥 Export Data")
    
    if st.button("Download Filtered Data", type="primary", use_container_width=True):
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv_data,
            file_name=f"shot_analysis_{team_filter.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # sidebar info
    st.markdown("---")
    st.markdown("### ℹ️ Dashboard Info")
    st.markdown(f"""
    <div style="background: #f1f5f9; padding: 1rem; border-radius: 8px; font-size: 0.85rem;">
        <strong>Data Summary:</strong><br>
        • {len(df):,} total shots in dataset<br>
        • {len(filtered_df):,} shots after filtering<br>
        • {len(df['team.name'].unique())} teams analyzed<br>
        • {len(df['shot.outcome.name'].unique())} outcome types
    </div>
    """, unsafe_allow_html=True)

