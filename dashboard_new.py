# import streamlit as st
# import pandas as pd
# import numpy as np
# from google.cloud import bigquery
# from google.oauth2 import service_account
# import os
# import plotly.graph_objects as go
#
# # ============================================================================
# # CONFIGURATION & STYLING (Pink Tags & Layout)
# # ============================================================================
# st.set_page_config(page_title="Third Eye | Performance Matrix", layout="wide")
#
# st.markdown(
#     """
#     <style>
#     span[data-baseweb="tag"] { background-color: #ffafcc !important; border-radius: 5px !important; }
#     span[data-baseweb="tag"] span { color: white !important; }
#     span[data-baseweb="tag"] svg { fill: white !important; }
#
#     .main-header { font-size: 32px; font-weight: bold; margin-bottom: 20px; display: flex; align-items: center; }
#     .summary-title {
#         background-color: #4A90E2;
#         color: white;
#         padding: 5px 15px;
#         border-radius: 3px;
#         font-weight: bold;
#         display: inline-block;
#         margin-top: 20px;
#         margin-bottom: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
#
# MY_COLORS = ['#cdb4db', '#ffc8dd', '#ffafcc', '#bde0fe', '#a2d2ff']
# PROJECT_ID = "mod-gcp-white-soi-dev-1"
# DATASET_ID = "mantak_database"
# TABLE_ID = "classified_predictions_third_eye"
# FULL_TABLE_PATH = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
# SPLIT_GROUP_TABLE = f"{PROJECT_ID}.{DATASET_ID}.split_groups_third_eye"
# SCENARIO_DETAILS = f"{PROJECT_ID}.{DATASET_ID}.scenario_details_third_eye"
#
# if 'selected_thresholds' not in st.session_state:
#     st.session_state.selected_thresholds = {}
#
#
# # ============================================================================
# # DATA FETCHING
# # ============================================================================
# @st.cache_resource
# def get_bq_client():
#     return bigquery.Client(project=PROJECT_ID)
#
#
# @st.cache_data
# def get_options(_client, col, table):
#     query = f"SELECT DISTINCT {col} FROM `{table}` WHERE {col} IS NOT NULL"
#     return _client.query(query).to_dataframe()[col].tolist()
#
#
# @st.cache_data(ttl=3600)
# def fetch_data(_client, models, split, version, datasets):
#     if not models: return pd.DataFrame()
#
#     # Explicit column naming to avoid KeyError
#     q = f"""
#         SELECT t1.frame, t1.confidence, t1.eval_type, t1.gt_id, t1.model as model_name,
#                t1.scenario, t1.class, t1.xmin, t1.ymin, t1.xmax, t1.ymax,
#                t3.dataset, t2.split, t2.version
#         FROM `{FULL_TABLE_PATH}` AS t1
#         LEFT JOIN `{SPLIT_GROUP_TABLE}` AS t2 ON t1.scenario = t2.scenario
#         LEFT JOIN `{SCENARIO_DETAILS}` AS t3 ON t1.scenario = t3.scenario
#         WHERE t1.model IN ('{"', '".join(models)}')
#     """
#     df = _client.query(q).to_dataframe()
#
#     if split != "All": df = df[df['split'].astype(str) == split]
#     if version != "All": df = df[df['version'].astype(str) == version]
#     if datasets and "All" not in datasets: df = df[df['dataset'].isin(datasets)]
#     return df
#
#
# @st.cache_data
# def fetch_meta(_client):
#     return _client.query(f"SELECT scenario, cluster, amount_of_frames FROM `{SCENARIO_DETAILS}`").to_dataframe()
#
#
# # ============================================================================
# # METRICS LOGIC (FAR = (FP / Total Frames) * 300)
# # ============================================================================
# def calculate_all_metrics(m_df, ann_df, meta_df, d_h, fpiou_h):
#     p_df = m_df.copy()
#     p_df['eval_type'] = p_df['eval_type'].replace({'D': d_h, 'FP_IOU': fpiou_h})
#     p_df = p_df[p_df['eval_type'] != 'Ignore']
#
#     scens = p_df['scenario'].unique()
#     m_sub = meta_df[meta_df['scenario'].isin(scens)]
#     total_f = m_sub['amount_of_frames'].sum()
#     total_c = m_sub['cluster'].nunique()
#
#     # GT Logic for Recall
#     gt = ann_df[ann_df['scenario'].isin(scens)].groupby(['scenario', 'id'])['frame'].nunique().reset_index(name='apps')
#
#     res = []
#     for t in np.arange(0.05, 1.0, 0.05):
#         sub = p_df[p_df['confidence'] >= t]
#         tps, fps = sub[sub['eval_type'] == 'TP'], sub[sub['eval_type'] == 'FP']
#
#         # EXACT FAR CALCULATION
#         far = (len(fps) / total_f * 300) if total_f > 0 else 0
#         prec = len(tps) / (len(tps) + len(fps)) if (len(tps) + len(fps)) > 0 else 0
#
#         matches = tps.groupby(['scenario', 'gt_id']).size().reset_index(name='tp_amt')
#         merged = gt.merge(matches, left_on=['scenario', 'id'], right_on=['scenario', 'gt_id'], how='left').fillna(0)
#         rec = len(tps) / (len(tps) + len(merged[merged['tp_amt'] == 0])) if (len(tps) + len(
#             merged[merged['tp_amt'] == 0])) > 0 else 0
#
#         # Track Metrics
#         tr_obs = (len(merged[merged['tp_amt'] / merged['apps'] >= 0.5]) / len(merged) * 100) if len(merged) > 0 else 0
#         tr_alert = (len(merged[merged['tp_amt'] > 0]) / len(merged) * 100) if len(merged) > 0 else 0
#
#         # Unique Track FP
#         unique_fp = len(fps.merge(m_sub[['scenario', 'cluster']], on='scenario').drop_duplicates(['cluster', 'class']))
#         tr_far = (unique_fp / total_c) if total_c > 0 else 0
#
#         res.append({
#             'threshold': round(t, 2), 'Detection Recall': rec, 'False Alarm Rate': far,
#             'Detection Precision': prec, 'Detection f1': (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0),
#             'Track recall observation': tr_obs, 'Track recall alerts': tr_alert, 'Track FAR': tr_far,
#             'Amount of Track FP': unique_fp, 'Avg TPs Confidence': tps['confidence'].mean() if not tps.empty else 0,
#             'TP': len(tps), 'FP': len(fps), 'FN': len(merged[merged['tp_amt'] == 0])
#         })
#     return pd.DataFrame(res)
#
#
# # ============================================================================
# # APP UI
# # ============================================================================
# client = get_bq_client()
# meta = fetch_meta(client)
# # Local CSV for annotations
# ann = pd.read_csv('annotations_third_eye.csv') if os.path.exists('annotations_third_eye.csv') else pd.DataFrame()
#
# # --- SIDEBAR ---
# st.sidebar.title("Settings")
# all_m = get_options(client, "model", FULL_TABLE_PATH)
# sel_m = st.sidebar.multiselect("Select Models", all_m, default=all_m[:2])
#
# st.sidebar.subheader("Dataset Filters")
# sel_s = st.sidebar.selectbox("Select Split",
#                              ["All"] + [str(x) for x in get_options(client, "split", SPLIT_GROUP_TABLE)])
# sel_v = st.sidebar.selectbox("Select Version",
#                              ["All"] + [str(x) for x in get_options(client, "version", SPLIT_GROUP_TABLE)])
# sel_d = st.sidebar.multiselect("Select DataSets", ["All"] + get_options(client, "dataset", SCENARIO_DETAILS),
#                                default=["All"])
#
# st.sidebar.subheader("Advanced Handling")
# d_h = st.sidebar.selectbox("Handle 'D' as:", ["Ignore", "TP", "FP"], index=2)
# f_h = st.sidebar.selectbox("Handle 'FP_IOU' as:", ["Ignore", "FP", "TP"], index=1)
#
# if sel_m and not ann.empty:
#     st.markdown('<div class="main-header">👁️ Third Eye | Performance Matrix</div>', unsafe_allow_html=True)
#
#     data = fetch_data(client, sel_m, sel_s, sel_v, sel_d)
#     roc_fig = go.Figure()
#     all_res = {}
#
#     for i, m in enumerate(sel_m):
#         m_res = calculate_all_metrics(data[data['model_name'] == m], ann, meta, d_h, f_h)
#         all_res[m] = m_res
#         color = MY_COLORS[i % len(MY_COLORS)]
#
#         # Main ROC Line with Legendgroup for coordination
#         roc_fig.add_trace(go.Scatter(
#             x=m_res['False Alarm Rate'], y=m_res['Detection Recall'],
#             name=m, mode='lines+markers', line=dict(color=color),
#             legendgroup=m,
#             customdata=m_res[['threshold', 'False Alarm Rate', 'Detection Recall']].values,
#             hovertemplate="<b>" + m + "</b><br>Confidence: %{customdata[0]:.2f}<br>FAR: %{customdata[1]:.2f}<br>Recall: %{customdata[2]:.1%}<extra></extra>"
#         ))
#
#         # Highlight Selected Point
#         t = st.session_state.selected_thresholds.get(m, 0.5)
#         row = m_res.iloc[(m_res['threshold'] - t).abs().idxmin()]
#         roc_fig.add_trace(go.Scatter(
#             x=[row['False Alarm Rate']], y=[row['Detection Recall']],
#             mode='markers', marker=dict(size=18, color=color, line=dict(width=2, color='white')),
#             showlegend=False, legendgroup=m, hoverinfo='skip'
#         ))
#
#     roc_fig.update_layout(
#         xaxis_title="False Alarm Rate (Normalized to 300 frames)",
#         yaxis_title="Recall", height=500,
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#     )
#
#     # Interactive selection with Crash protection
#     ev = st.plotly_chart(roc_fig, use_container_width=True, on_select="rerun")
#     if ev and "selection" in ev and len(ev["selection"]["points"]) > 0:
#         point = ev["selection"]["points"][0]
#         if "customdata" in point and point["customdata"] is not None:
#             model_idx = point["curve_number"] // 2
#             if model_idx < len(sel_m):
#                 st.session_state.selected_thresholds[sel_m[model_idx]] = point["customdata"][0]
#                 st.rerun()
#
#     # --- SUMMARY TABLE ---
#     st.markdown('<div class="summary-title">📋 Comparison Summary</div>', unsafe_allow_html=True)
#     summary_rows = []
#     for m in sel_m:
#         t = st.session_state.selected_thresholds.get(m, 0.5)
#         r = all_res[m].iloc[(all_res[m]['threshold'] - t).abs().idxmin()].to_dict()
#         r['Model'] = m
#         summary_rows.append(r)
#
#     comp_df = pd.DataFrame(summary_rows)
#
#
#     def color_metric(val):
#         if not isinstance(val, (int, float)): return ''
#         if val > 0.8: return 'background-color: #90EE90'
#         if val < 0.4: return 'background-color: #FF0000; color: white'
#         return ''
#
#
#     styled_df = comp_df[['Model', 'Detection Recall', 'False Alarm Rate', 'Detection Precision', 'Detection f1',
#                          'Track recall observation', 'Track recall alerts', 'Track FAR', 'Amount of Track FP',
#                          'Avg TPs Confidence', 'TP', 'FP', 'FN']].style.format({
#         'Detection Recall': '{:.1%}', 'Detection Precision': '{:.1%}', 'Detection f1': '{:.1%}',
#         'Track recall observation': '{:.1f}%', 'Track recall alerts': '{:.1f}%',
#         'False Alarm Rate': '{:.2f}', 'Track FAR': '{:.2f}', 'Avg TPs Confidence': '{:.2f}'
#     }).applymap(color_metric, subset=['Detection Recall', 'Detection Precision'])
#
#     st.dataframe(styled_df, use_container_width=True)
#
#     # --- PERFORMANCE CARDS ---
#     st.markdown("---")
#     cols = st.columns(len(sel_m))
#     best_val = comp_df['Detection f1'].max()
#
#     for i, m in enumerate(sel_m):
#         d = next(item for item in summary_rows if item['Model'] == m)
#         is_best = d['Detection f1'] == best_val
#         with cols[i]:
#             st.markdown(f"""
#             <div style="padding:20px; border-radius:10px; border: {'2px solid #FF1493' if is_best else '1px solid #ddd'}; background-color: white; text-align:center;">
#                 <div style="float:right; font-size: 0.7em; background:#eee; padding:2px 5px; border-radius:3px;">Th: {st.session_state.selected_thresholds.get(m, 0.5):.2f}</div>
#                 <h4 style="margin-bottom:10px;">{m} {'🏆' if is_best else ''}</h4>
#                 <h2 style="color: #FF1493; margin:0;">{d['Detection f1']:.1%}</h2>
#                 <p style="font-size:0.85em; color:#666; margin-top:10px;">FAR: {d['False Alarm Rate']:.2f} | TP: {d['TP']}</p>
#             </div>
#             """, unsafe_allow_html=True)
# else:
#     st.info("Please select models in the sidebar and ensure annotations are available.")



import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(page_title="Third Eye | Interactive ROC", layout="wide")

# CSS Injection לשינוי צבע התיגיות ב-Multiselect לורוד
st.markdown(
    """
    <style>
    span[data-baseweb="tag"] {
        background-color: #ffafcc !important;
        border-radius: 5px !important;
    }
    span[data-baseweb="tag"] span {
        color: white !important;
    }
    span[data-baseweb="tag"] svg {
        fill: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    if split != "All": where_clauses.append(f"CAST(t2.split AS STRING) = '{split}'")
    if version != "All": where_clauses.append(f"CAST(t2.version AS STRING) = '{version}'")
    if selected_datasets and "All" not in selected_datasets:
        ds_str = "', '".join(selected_datasets)
        where_clauses.append(f"CAST(t3.dataset AS STRING) IN ('{ds_str}')")

    query = f"""SELECT t1.frame, t1.confidence, t1.eval_type, t1.gt_id, t1.model as model_name, t1.scenario
                FROM `{FULL_TABLE_PATH}` AS t1
                LEFT JOIN `{SPLIT_GROUP_TABLE}` AS t2 ON t1.scenario = t2.scenario
                LEFT JOIN `{SCENARIO_DETAILS}` AS t3 ON t1.scenario = t3.scenario
                WHERE {" AND ".join(where_clauses)}"""

    df = _client.query(query).to_dataframe()
    df['gt_key'] = df['scenario'].astype(str) + "_" + df['frame'].astype(str) + "_" + df['gt_id'].astype(str)
    return df


@st.cache_data
def load_annotations():
    if os.path.exists('annotations_third_eye.csv'):
        ann_df = pd.read_csv('annotations_third_eye.csv')
        ann_df['gt_key'] = ann_df['scenario'].astype(str) + "_" + ann_df['frame'].astype(str) + "_" + ann_df[
            'id'].astype(str)
        return ann_df
    return pd.DataFrame()


# ============================================================================
# CALCULATION (FAR = (FP / amount_of_frames) * 300)
# ============================================================================

def calculate_advanced_metrics(model_df, ann_df, total_gt_frames, d_handling, fpiou_handling, total_frames_from_db):
    processed_df = model_df.copy()
    mapping = {'D': d_handling, 'FP_IOU': fpiou_handling}
    processed_df['effective_type'] = processed_df['eval_type'].replace(mapping)
    processed_df = processed_df[processed_df['effective_type'] != 'Ignore']

    thresholds = np.arange(0.05, 1.0, 0.05)
    results = []
    fp_base = processed_df[processed_df['effective_type'] == 'FP']

    for t in thresholds:
        subset = processed_df[processed_df['confidence'] >= t]
        tp_subset = subset[subset['effective_type'] == 'TP']

        fp_amount = len(fp_base[fp_base['confidence'] >= t])
        far_val = (fp_amount / total_frames_from_db * 300) if total_frames_from_db > 0 else 0

        u_tp_frames = tp_subset['gt_key'].nunique()
        tp_count = len(tp_subset)

        det_recall = u_tp_frames / total_gt_frames if total_gt_frames > 0 else 0
        det_precision = tp_count / (tp_count + fp_amount) if (tp_count + fp_amount) > 0 else 0
        f1 = 2 * (det_precision * det_recall) / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0

        results.append({
            'threshold': round(t, 2),
            'Detection Recall': det_recall,
            'False Alarm Rate': far_val,
            'Detection Precision': det_precision,
            'Detection f1': f1,
            'TP': int(tp_count), 'FP': int(fp_amount), 'FN': int(total_gt_frames - u_tp_frames)
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

        if not full_df.empty:
            relevant_scenarios = full_df['scenario'].unique().tolist()
            scenarios_str = "', '".join(relevant_scenarios)
            frames_query = f"SELECT SUM(amount_of_frames) as total FROM `{SCENARIO_DETAILS}` WHERE scenario IN ('{scenarios_str}')"
            total_frames_db = client.query(frames_query).to_dataframe()['total'].iloc[0] or 0

            filtered_ann = ann_df[ann_df['scenario'].isin(relevant_scenarios)]
            total_gt_frames = filtered_ann['gt_key'].nunique()

            roc_fig = go.Figure()
            all_metrics = {}

            for i, m in enumerate(selected_models):
                m_df = full_df[full_df['model_name'] == m]
                res_df = calculate_advanced_metrics(m_df, filtered_ann, total_gt_frames, d_h, fpiou_h, total_frames_db)
                all_metrics[m], color = res_df, MY_COLORS[i % len(MY_COLORS)]

                roc_fig.add_trace(go.Scatter(
                    x=res_df['False Alarm Rate'], y=res_df['Detection Recall'],
                    name=m, mode='lines+markers', line=dict(color=color, width=3),
                    marker=dict(size=10, opacity=0.5), customdata=res_df['threshold'],
                    hovertemplate="<b>%{fullData.name}</b><br>Conf: %{customdata}<br>FAR: %{x:.2f}<br>Rec: %{y:.1%}<extra></extra>"
                ))

                curr_t = st.session_state.selected_thresholds.get(m, 0.5)
                sel_row = res_df.iloc[(res_df['threshold'] - curr_t).abs().idxmin()]
                roc_fig.add_trace(go.Scatter(x=[sel_row['False Alarm Rate']], y=[sel_row['Detection Recall']],
                                             mode='markers', marker=dict(color=color, size=16, symbol='circle',
                                                                         line=dict(color='white', width=2)),
                                             showlegend=False))

            roc_fig.update_layout(xaxis_title="False Alarm Rate (Normalized)", yaxis_title="Recall",
                                  template='plotly_white', clickmode='event+select')
            event = st.plotly_chart(roc_fig, use_container_width=True, on_select="rerun")

            if event and "selection" in event and len(event["selection"]["points"]) > 0:
                p = event["selection"]["points"][0]
                m_idx = p["curve_number"] // 2
                if m_idx < len(selected_models):
                    st.session_state.selected_thresholds[selected_models[m_idx]] = p["customdata"]
                    st.rerun()

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
                ['Model', 'Detection Recall', 'False Alarm Rate', 'Detection Precision', 'Detection f1', 'TP', 'FP',
                 'FN']].style
            .apply(lambda r: ['color: #FF1493; font-weight: bold' if r['Model'] == best_model else '' for _ in r],
                   axis=1)
            .map(style_metric_values, subset=['Detection Recall', 'Detection Precision'])
            .format({
                'Detection Recall': '{:.2%}', 'Detection Precision': '{:.2%}', 'Detection f1': '{:.2%}',
                'False Alarm Rate': '{:.2f}', 'TP': '{:.0f}', 'FP': '{:.0f}', 'FN': '{:.0f}'
            }), use_container_width=True)

            st.markdown("---")
            card_cols = st.columns(len(selected_models))
            for i, m in enumerate(selected_models):
                data = next(item for item in comp_list if item['Model'] == m)
                curr_t = st.session_state.selected_thresholds.get(m, 0.5)
                bg = "#fff0f6" if m == best_model else "#ffffff"
                border = "#FF1493" if m == best_model else "#ddd"

                with card_cols[i]:
                    html_card = f"""<div style="padding:15px; border-radius:10px; border: 1px solid {border}; background-color: {bg}; text-align:center; position: relative; min-height: 150px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
<div style="position: absolute; top: 10px; right: 10px; background-color: #f0f2f6; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; border: 1px solid #ccc; font-weight: bold; color: #333;">Th: {curr_t:.2f}</div>
<h4 style="margin-top: 25px; margin-bottom: 10px; font-size: 1.1em; color: #333;">{m} {'🏆' if m == best_model else ''}</h4>
<p style="font-size:1.6em; font-weight:bold; color:#FF1493; margin: 15px 0 5px 0;">{data['Detection f1']:.1%}</p>
<p style="font-size:0.85em; margin:0; color: #555;">FAR: <b>{data['False Alarm Rate']:.2f}</b> | TP: <b>{data['TP']:.0f}</b></p>
</div>"""
                    st.markdown(html_card, unsafe_allow_html=True)
else:
    st.info("Please select models to begin.")