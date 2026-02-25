import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ID = "mod-gcp-white-soi-dev-1"
DATASET_ID = "mantak_database"
TABLE_ID = "classified_predictions_third_eye"
FULL_TABLE_PATH = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Configuration for model column name
# Change this if your table uses a different column name
MODEL_COLUMN_NAME = "model"  # Your table uses: model


@st.cache_resource
def get_bigquery_client():
    try:
        key_path = "service-account-key.json"
        if os.path.exists(key_path):
            credentials = service_account.Credentials.from_service_account_file(key_path)
            return bigquery.Client(credentials=credentials, project=PROJECT_ID)
        return bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        st.error(f"Failed to create BigQuery client: {str(e)}")
        return None


@st.cache_data(ttl=600)
def load_initial_data(_client, model_column):
    """Load all data from BigQuery with configurable model column"""
    # Try to load with model column first
    query_with_model = f"""
        SELECT frame, confidence, eval_type, gt_id, {model_column}
        FROM `{FULL_TABLE_PATH}`
        WHERE confidence >= 0.0
        ORDER BY confidence DESC
        LIMIT 500000
    """
    
    try:
        df = _client.query(query_with_model).to_dataframe()
        df = df.rename(columns={model_column: 'model_name'})
        return df, True
    except Exception as e:
        # If model column doesn't exist, load without it
        st.warning(f"⚠️ Column '{model_column}' not found. Loading data without model filtering.")
        query_without_model = f"""
            SELECT frame, confidence, eval_type, gt_id
            FROM `{FULL_TABLE_PATH}`
            WHERE confidence >= 0.0
            ORDER BY confidence DESC
            LIMIT 500000
        """
        df = _client.query(query_without_model).to_dataframe()
        return df, False


def calculate_roc_for_model(df, model_name=None):
    """Calculate ROC curve points for a specific model or all models"""
    # Filter by model if specified
    if model_name and model_name != "All Models" and 'model_name' in df.columns:
        df_filtered = df[df['model_name'] == model_name].copy()
    else:
        df_filtered = df.copy()
    
    if len(df_filtered) == 0:
        return pd.DataFrame(), 0, 0
    
    # Basic statistics
    num_frames = df_filtered['frame'].nunique()
    total_gt = df_filtered['gt_id'].nunique()

    # Pre-sort confidence values for faster filtering
    df_filtered = df_filtered.sort_values('confidence', ascending=False)

    # Pre-calculate ROC curve points
    thresholds = np.linspace(0, 1, 101)  # More points for smoother curve
    roc_points = []

    # Vectorized computation where possible
    is_tp = df_filtered['eval_type'] == 'TP'
    is_fp = df_filtered['eval_type'] == 'FP'

    for t in thresholds:
        # Use boolean indexing for faster filtering
        above_threshold = df_filtered['confidence'] >= t
        temp_filtered_tp = above_threshold & is_tp
        temp_filtered_fp = above_threshold & is_fp

        tp_count = temp_filtered_tp.sum()
        fp_count = temp_filtered_fp.sum()
        detected_gt = df_filtered.loc[temp_filtered_tp, 'gt_id'].nunique()

        roc_points.append({
            'threshold': round(t, 3),
            'recall': detected_gt / total_gt if total_gt > 0 else 0,
            'far': fp_count / num_frames if num_frames > 0 else 0,
            'tp': tp_count,
            'fp': fp_count
        })

    return pd.DataFrame(roc_points), total_gt, num_frames


def create_roc_plot(roc_df, current_metrics):
    """Create interactive ROC curve using Plotly"""
    fig = go.Figure()
    
    # ROC curve line
    fig.add_trace(go.Scatter(
        x=roc_df['far'],
        y=roc_df['recall'],
        mode='lines+markers',
        name='ROC Curve',
        line=dict(color='#db2777', width=2),
        marker=dict(size=4, color=roc_df['threshold'], 
                   colorscale='Reds', showscale=True,
                   colorbar=dict(title="Threshold")),
        hovertemplate='<b>Threshold:</b> %{marker.color:.3f}<br>' +
                      '<b>FAR:</b> %{x:.4f}<br>' +
                      '<b>Recall:</b> %{y:.2%}<br>' +
                      '<extra></extra>'
    ))
    
    # Current selected point
    fig.add_trace(go.Scatter(
        x=[current_metrics['far']],
        y=[current_metrics['recall']],
        mode='markers',
        name='Selected Point',
        marker=dict(size=15, color='red', 
                   line=dict(color='black', width=2)),
        hovertemplate='<b>Selected Threshold:</b> %.3f<br>' % current_metrics['threshold'] +
                      '<b>FAR:</b> %{x:.4f}<br>' +
                      '<b>Recall:</b> %{y:.2%}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='ROC Curve: Recall vs False Alarm Rate',
        xaxis_title='False Alarm Rate (FAR)',
        yaxis_title='Recall',
        hovermode='closest',
        showlegend=True,
        height=500
    )
    
    return fig


# ============================================================================
# UI & DASHBOARD
# ============================================================================
st.set_page_config(page_title="Third Eye Model Analysis", page_icon="📊", layout="wide")

st.title("📊 Model Evaluation Dashboard - Third Eye")

client = get_bigquery_client()

if client is None:
    st.error("❌ Could not connect to BigQuery. Please check your credentials.")
    st.stop()

try:
    # Load data (cached)
    with st.spinner("Loading data from BigQuery..."):
        df, has_model_column = load_initial_data(client, MODEL_COLUMN_NAME)
    
    # Get list of unique models
    if has_model_column and 'model_name' in df.columns:
        available_models = ["All Models"] + sorted(df['model_name'].unique().tolist())
    else:
        available_models = ["All Models"]
        has_model_column = False
    
    st.success(f"✅ Data loaded successfully! ({len(df):,} predictions)")

    # --- Sidebar ---
    st.sidebar.header("📋 Settings")
    
    # Model selector
    if has_model_column and len(available_models) > 2:  # More than just "All Models"
        st.sidebar.markdown("### 🤖 Model Selection")
        selected_model = st.sidebar.selectbox(
            "Select model to analyze",
            options=available_models,
            index=0,
            help="Choose a specific model or view all models combined"
        )
        
        # Show model statistics
        with st.sidebar.expander("📊 Model Statistics"):
            for model in available_models[1:]:  # Skip "All Models"
                count = len(df[df['model_name'] == model])
                st.write(f"**{model}:** {count:,} predictions")
    else:
        selected_model = "All Models"
    
    # Calculate ROC for selected model
    with st.spinner(f"Calculating ROC curve for {selected_model}..."):
        roc_df, total_gt, num_frames = calculate_roc_for_model(df, selected_model)
    
    if len(roc_df) == 0:
        st.error(f"❌ No data available for model: {selected_model}")
        st.stop()
    
    # Confidence threshold slider
    st.sidebar.markdown("### 🎚️ Confidence Threshold")
    selected_threshold = st.sidebar.slider(
        "Select confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Adjust the confidence threshold to see how metrics change"
    )

    # Find the closest pre-calculated point
    idx = (roc_df['threshold'] - selected_threshold).abs().argsort()[0]
    current_metrics = roc_df.iloc[idx]

    # --- Display Metrics ---
    st.markdown("### 📈 Current Metrics")
    if has_model_column:
        st.caption(f"**Model:** {selected_model} | **Threshold:** {current_metrics['threshold']:.3f}")
    else:
        st.caption(f"**Threshold:** {current_metrics['threshold']:.3f}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Recall", 
            f"{current_metrics['recall']:.2%}",
            help="Percentage of ground truth objects detected"
        )
    
    with col2:
        st.metric(
            "⚠️ FAR (per frame)", 
            f"{current_metrics['far']:.4f}",
            help="False alarms per frame"
        )
    
    with col3:
        st.metric(
            "✅ True Positives", 
            int(current_metrics['tp']),
            help="Number of correct detections"
        )
    
    with col4:
        st.metric(
            "❌ False Positives", 
            int(current_metrics['fp']),
            help="Number of false alarms"
        )

    # --- ROC Curve ---
    st.markdown("---")
    st.markdown("### 📉 ROC Curve Analysis")
    
    col_graph, col_info = st.columns([3, 1])

    with col_graph:
        fig = create_roc_plot(roc_df, current_metrics)
        if has_model_column:
            fig.update_layout(title=f'ROC Curve: {selected_model}')
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("#### 📊 Dataset Info")
        if has_model_column:
            # Count predictions for selected model
            if selected_model != "All Models":
                model_preds = len(df[df['model_name'] == selected_model])
            else:
                model_preds = len(df)
            st.info(f"""
            **Model:** {selected_model}  
            **Total GT Objects:** {total_gt:,}  
            **Total Frames:** {num_frames:,}  
            **Predictions:** {model_preds:,}
            """)
        else:
            st.info(f"""
            **Total GT Objects:** {total_gt:,}  
            **Total Frames:** {num_frames:,}  
            **Total Predictions:** {len(df):,}
            """)
        
        st.markdown("#### 🔍 Current Selection")
        st.success(f"""
        **Threshold:** {current_metrics['threshold']:.3f}  
        **Recall:** {current_metrics['recall']:.2%}  
        **FAR:** {current_metrics['far']:.4f}
        """)

    # --- Compare Models (if multiple models available) ---
    if has_model_column and len(available_models) > 2:
        st.markdown("---")
        st.markdown("### 🔄 Compare Models")
        
        compare_models = st.checkbox("Show model comparison", value=False)
        
        if compare_models:
            st.info("Calculating ROC curves for all models...")
            
            comparison_fig = go.Figure()
            
            for model in available_models[1:]:  # Skip "All Models"
                model_roc_df, _, _ = calculate_roc_for_model(df, model)
                if len(model_roc_df) > 0:
                    comparison_fig.add_trace(go.Scatter(
                        x=model_roc_df['far'],
                        y=model_roc_df['recall'],
                        mode='lines+markers',
                        name=model,
                        marker=dict(size=3),
                        line=dict(width=2),
                        hovertemplate=f'<b>{model}</b><br>' +
                                    'Threshold: %{text}<br>' +
                                    'FAR: %{x:.4f}<br>' +
                                    'Recall: %{y:.2%}<br>' +
                                    '<extra></extra>',
                        text=model_roc_df['threshold']
                    ))
            
            comparison_fig.update_layout(
                title='Model Comparison: ROC Curves',
                xaxis_title='False Alarm Rate (FAR)',
                yaxis_title='Recall',
                hovermode='closest',
                showlegend=True,
                height=600
            )
            
            st.plotly_chart(comparison_fig, use_container_width=True)

    # --- Data Preview ---
    st.markdown("---")
    st.markdown("### 📋 Data Preview")
    
    show_sample = st.checkbox("Show filtered predictions sample", value=False)
    
    if show_sample:
        # Filter by model and threshold
        if selected_model != "All Models" and has_model_column:
            filtered_df = df[(df['confidence'] >= selected_threshold) & (df['model_name'] == selected_model)]
        else:
            filtered_df = df[df['confidence'] >= selected_threshold]
        
        st.write(f"**Showing top 100 predictions with confidence >= {selected_threshold:.2f}**")
        if has_model_column:
            st.write(f"**Model:** {selected_model}")
        st.write(f"Total filtered predictions: {len(filtered_df):,}")
        st.dataframe(filtered_df.head(100), use_container_width=True)

    # --- ROC Data Table ---
    show_roc_data = st.checkbox("Show ROC curve data table", value=False)
    
    if show_roc_data:
        st.write(f"**Pre-calculated ROC points for {selected_model}:**")
        st.dataframe(roc_df, use_container_width=True)
        
        # Download button for ROC data
        csv = roc_df.to_csv(index=False)
        st.download_button(
            label="📥 Download ROC data as CSV",
            data=csv,
            file_name=f"roc_data_{selected_model.replace(' ', '_')}.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.exception(e)
