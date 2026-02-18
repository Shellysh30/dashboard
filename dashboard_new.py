import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Third Eye | Interactive ROC", layout="wide")

FPS = 30
MY_COLORS = ['#cdb4db', '#ffc8dd', '#ffafcc', '#bde0fe', '#a2d2ff']

PROJECT_ID = "mod-gcp-white-soi-dev-1"
DATASET_ID = "mantak_database"
TABLE_ID = "classified_predictions_third_eye"
FULL_TABLE_PATH = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
SPLIT_GROUP_TABLE = f"{PROJECT_ID}.{DATASET_ID}.split_groups_third_eye"
SCENARIO_DETAILS = f"{PROJECT_ID}.{DATASET_ID}.scenario_details_third_eye"

if 'selected_thresholds' not in st.session_state:
    st.session_state.selected_thresholds = {}


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

@st.cache_resource
def get_bigquery_client():
    key_path = "service-account-key.json"
    if os.path.exists(key_path):
        credentials = service_account.Credentials.from_service_account_file(key_path)
        return bigquery.Client(credentials=credentials, project=PROJECT_ID)
    return bigquery.Client(project=PROJECT_ID)


@st.cache_data
def get_filter_options(_client, column_name, table_path):
    query = f"SELECT DISTINCT {column_name} FROM `{table_path}` WHERE {column_name} IS NOT NULL"
    return _client.query(query).to_dataframe()[column_name].tolist()


@st.cache_data(ttl=3600)
def fetch_model_data(_client, selected_models, split="All", version="All", selected_datasets=["All"]):
    if not selected_models: return pd.DataFrame()
    models_str = "', '".join(selected_models)

    where_clauses = [f"t1.model IN ('{models_str}')"]
    if split != "All": where_clauses.append(f"t2.split = '{split}'")
    if version != "All": where_clauses.append(f"CAST(t2.version AS STRING) = '{version}'")
    if selected_datasets and "All" not in selected_datasets:
        ds_str = "', '".join(selected_datasets)
        where_clauses.append(f"t3.dataset IN ('{ds_str}')")

    query = f"""SELECT t1.frame, t1.confidence, t1.eval_type, t1.gt_id, t1.model as model_name, t1.scenario
                FROM `{FULL_TABLE_PATH}` AS t1
                LEFT JOIN `{SPLIT_GROUP_TABLE}` AS t2 ON t1.scenario = t2.scenario
                LEFT JOIN `{SCENARIO_DETAILS}` AS t3 ON t1.scenario = t3.scenario
                WHERE {" AND ".join(where_clauses)}"""

    df = _client.query(query).to_dataframe()
    df['gt_key'] = df['scenario'].astype(str) + "_" + df['frame'].astype(str) + "_" + df['gt_id'].astype(str)
    df['track_key'] = df['scenario'].astype(str) + "_" + df['gt_id'].astype(str)
    return df


@st.cache_data
def load_annotations():
    if os.path.exists('annotations_third_eye.csv'):
        ann_df = pd.read_csv('annotations_third_eye.csv')
        ann_df['gt_key'] = ann_df['scenario'].astype(str) + "_" + ann_df['frame'].astype(str) + "_" + ann_df[
            'id'].astype(str)
        ann_df['track_key'] = ann_df['scenario'].astype(str) + "_" + ann_df['id'].astype(str)
        return ann_df
    return pd.DataFrame()


# ============================================================================
# OPTIMIZED CALCULATION
# ============================================================================

def calculate_advanced_metrics(model_df, ann_df, total_gt_frames, total_gt_tracks, d_handling, fpiou_handling):
    actual_num_frames = ann_df['frame'].nunique()
    total_minutes = max((actual_num_frames / FPS) / 60, 1.0)

    processed_df = model_df.copy()
    mapping = {'D': d_handling, 'FP_IOU': fpiou_handling}
    processed_df['effective_type'] = processed_df['eval_type'].replace(mapping)
    processed_df = processed_df[processed_df['effective_type'] != 'Ignore']

    thresholds = np.arange(0.05, 1.0, 0.05)
    results = []
    fp_base = processed_df[processed_df['effective_type'] == 'FP'].sort_values(['scenario', 'frame'])

    for t in thresholds:
        subset = processed_df[processed_df['confidence'] >= t]
        tp_subset = subset[subset['effective_type'] == 'TP']
        current_fp = fp_base[fp_base['confidence'] >= t]

        if not current_fp.empty:
            unique_fp_events = (current_fp.groupby('scenario')['frame'].diff() > 150).sum() + current_fp[
                'scenario'].nunique()
        else:
            unique_fp_events = 0

        far_val = unique_fp_events / total_minutes
        u_tp_frames = tp_subset['gt_key'].nunique()
        tp_count, fp_count = len(tp_subset), len(current_fp)

        det_recall = u_tp_frames / total_gt_frames if total_gt_frames > 0 else 0
        det_precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        f1 = 2 * (det_precision * det_recall) / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0

        results.append({
            'threshold': round(t, 2), 'Detection Recall': det_recall, 'False Alarm Rate (per minute)': far_val,
            'Detection Precision': det_precision, 'Detection f1': f1,
            'TP': int(tp_count), 'FP': int(fp_count), 'FN': int(total_gt_frames - u_tp_frames)
        })
    return pd.DataFrame(results)


# ============================================================================
# APP UI
# ============================================================================
st.title("👁️ Third Eye | Performance Matrix")

client = get_bigquery_client()
ann_df = load_annotations()

st.sidebar.header("Settings")
all_models = get_filter_options(client, "model", FULL_TABLE_PATH)
selected_models = st.sidebar.multiselect("Select Models", all_models, max_selections=5)

st.sidebar.subheader("Dataset Filters")
split_opts = ["All"] + get_filter_options(client, "split", SPLIT_GROUP_TABLE)
version_opts = ["All"] + get_filter_options(client, "version", SPLIT_GROUP_TABLE)
dataset_opts = get_filter_options(client, "dataset", SCENARIO_DETAILS)

selected_split = st.sidebar.selectbox("Select Split", split_opts)
selected_version = st.sidebar.selectbox("Select Version", version_opts)
selected_datasets = st.sidebar.multiselect("Select DataSets", ["All"] + dataset_opts, default=["All"])
if not selected_datasets: selected_datasets = ["All"]

st.sidebar.subheader("Advanced Handling")
d_h = st.sidebar.selectbox("Handle 'D' as:", ["Ignore", "TP", "FP"], index=0)
fpiou_h = st.sidebar.selectbox("Handle 'FP_IOU' as:", ["Ignore", "FP", "TP"], index=1)

if selected_models and not ann_df.empty:
    with st.spinner("Processing..."):
        full_df = fetch_model_data(client, selected_models, selected_split, selected_version, selected_datasets)
        relevant_scenarios = full_df['scenario'].unique()
        filtered_ann = ann_df[ann_df['scenario'].isin(relevant_scenarios)]
        total_gt_frames, total_gt_tracks = filtered_ann['gt_key'].nunique(), filtered_ann['track_key'].nunique()

    if not full_df.empty:
        roc_fig = go.Figure()
        all_metrics = {}

        for i, m in enumerate(selected_models):
            m_df = full_df[full_df['model_name'] == m]
            res_df = calculate_advanced_metrics(m_df, filtered_ann, total_gt_frames, total_gt_tracks, d_h, fpiou_h)
            all_metrics[m], color = res_df, MY_COLORS[i % len(MY_COLORS)]

            roc_fig.add_trace(go.Scatter(
                x=res_df['False Alarm Rate (per minute)'], y=res_df['Detection Recall'],
                name=m, mode='lines+markers', line=dict(color=color, width=3),
                marker=dict(size=8, opacity=0.3), customdata=res_df['threshold'],
                hovertemplate="<b>%{fullData.name}</b><br>Conf: %{customdata}<br>FAR: %{x:.1f}<br>Rec: %{y:.1%}<extra></extra>"
            ))

            curr_t = st.session_state.selected_thresholds.get(m, 0.5)
            sel_row = res_df.iloc[(res_df['threshold'] - curr_t).abs().idxmin()]
            roc_fig.add_trace(go.Scatter(x=[sel_row['False Alarm Rate (per minute)']], y=[sel_row['Detection Recall']],
                                         mode='markers', marker=dict(color=color, size=15, symbol='circle',
                                                                     line=dict(color='white', width=2)),
                                         showlegend=False))

        roc_fig.update_layout(xaxis_title="FAR (per minute)", yaxis_title="Recall", template='plotly_white',
                              clickmode='event+select')
        event = st.plotly_chart(roc_fig, use_container_width=True, on_select="rerun")

        if event and "selection" in event and len(event["selection"]["points"]) > 0:
            p = event["selection"]["points"][0]
            m_idx = p["curve_number"] // 2
            if m_idx < len(selected_models):
                st.session_state.selected_thresholds[selected_models[m_idx]] = p["customdata"]
                st.rerun()

        # --- SUMMARY TABLE WITH STYLING AND INTEGER FORMATTING ---
        st.subheader("📋 Comparison Summary")
        comp_list = []
        for m in selected_models:
            df = all_metrics[m]
            t = st.session_state.selected_thresholds.get(m, 0.5)
            row = df.iloc[(df['threshold'] - t).abs().idxmin()].to_dict()
            row['Model'] = m
            comp_list.append(row)

        comp_df = pd.DataFrame(comp_list)
        best_model = comp_df.loc[comp_df['Detection f1'].idxmax(), 'Model']


        def style_metric_values(val):
            if isinstance(val, (int, float)):
                if val > 0.80:
                    return 'background-color: #9ADE7B; color: black'
                elif val < 0.30:
                    return 'background-color: #FF0000; color: white'
            return ''


        st.dataframe(comp_df[
            ['Model', 'Detection Recall', 'False Alarm Rate (per minute)', 'Detection Precision', 'Detection f1', 'TP',
             'FP', 'FN']].style
        .apply(lambda r: ['color: #FF1493; font-weight: bold' if r['Model'] == best_model else '' for _ in r], axis=1)
        .map(style_metric_values, subset=['Detection Recall', 'Detection Precision'])
        .format({
            'Detection Recall': '{:.1%}',
            'Detection Precision': '{:.2%}',
            'Detection f1': '{:.1%}',
            'False Alarm Rate (per minute)': '{:.2f}',
            'TP': '{:.0f}',
            'FP': '{:.0f}',
            'FN': '{:.0f}'
        }), use_container_width=True)

        # CARDS SECTION
        st.markdown("---")
        card_cols = st.columns(len(selected_models))
        for i, m in enumerate(selected_models):
            data = next(item for item in comp_list if item['Model'] == m)
            curr_t = st.session_state.selected_thresholds.get(m, 0.5)
            bg = "#fff0f6" if m == best_model else "#ffffff"
            border = "#FF1493" if m == best_model else "#ddd"


            with card_cols[i]:
                html_card = f"""<div style="padding:15px; border-radius:10px; border: 1px solid {border}; background-color: {bg}; text-align:center; position: relative; min-height: 150px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
<div style="position: absolute; top: 10px; right: 10px; background-color: #f0f2f6; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; border: 1px solid #ccc; font-weight: bold; color: #333;">Confidence Th: {curr_t:.2f}</div>
<h4 style="margin-top: 25px; margin-bottom: 10px; font-size: 1.1em; color: #333;">{m} {'🏆' if m == best_model else ''}</h4>
<p style="font-size:1.6em; font-weight:bold; color:#FF1493; margin: 15px 0 5px 0;">{data['Detection f1']:.1%}</p>
<p style="font-size:0.85em; margin:0; color: #555;">FAR: <b>{data['False Alarm Rate (per minute)']:.2f}</b> | TP: <b>{data['TP']:.0f}</b></p>
</div>"""
                st.markdown(html_card, unsafe_allow_html=True)
else:
    st.info("Please select models to begin.")

