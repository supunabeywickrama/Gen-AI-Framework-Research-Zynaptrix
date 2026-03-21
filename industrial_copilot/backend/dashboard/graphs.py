"""
graphs.py — Reusable Plotly components for the dashboard.
"""
import plotly.graph_objects as go
import pandas as pd

def create_sensor_gauge(value: float, sensor_name: str, config: dict) -> go.Figure:
    """
    Creates a Plotly Indicator (Gauge) for a single sensor.
    Maps the 'normal_range' to green and outer limits to red.
    """
    min_val, max_val = config.get("normal_range", [0, 100])
    
    # Estimate reasonable gauge bounds based on normal range
    span = max_val - min_val
    gauge_min = max(0, min_val - span)
    gauge_max = max_val + span
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': sensor_name.replace("_", " ").title(), 'font': {'size': 14}},
        number={'font': {'size': 20}},
        gauge={
            'axis': {'range': [gauge_min, gauge_max], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [gauge_min, min_val], 'color': "#ffcccb"},  # Red zone (low)
                {'range': [min_val, max_val], 'color': "#e0fce0"},    # Green zone (normal)
                {'range': [max_val, gauge_max], 'color': "#ffcccb"}   # Red zone (high)
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=200
    )
    return fig

def create_anomaly_score_chart(scores: list, threshold: float) -> go.Figure:
    """
    Creates a live-updating line chart plotting the anomaly score over time.
    """
    df = pd.DataFrame({"Time": range(len(scores)), "Score": scores})
    
    fig = go.Figure()
    
    # Plot score line
    fig.add_trace(go.Scatter(
        x=df["Time"], 
        y=df["Score"],
        mode='lines+markers',
        name='Anomaly Score',
        line=dict(color='royalblue', width=2),
        marker=dict(size=4)
    ))
    
    # Plot threshold line
    fig.add_hline(
        y=threshold, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Critical Threshold", 
        annotation_position="top left"
    )
    
    fig.update_layout(
        title="Live Anomaly Reconstruction Score",
        xaxis_title="Recent Readings",
        yaxis_title="Score",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white"
    )
    
    return fig
