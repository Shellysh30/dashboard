'''
    to open the streamlit dashboard run in the terminal:
            streamlit run .\dashboard_new.py
'''


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


st.markdown(
    """
    <style>
    span[data-baseweb="tag"] { background-color: #ffafcc !important; border-radius: 5px !important; }
    span[data-baseweb="tag"] span { color: white !important; }
    span[data-baseweb="tag"] svg { fill: white !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("👁️ Third Eye | Performance Matrix")

MY_COLORS = ['#cdb4db', '#ffc8dd', '#ffafcc', '#bde0fe', '#a2d2ff']
PROJECT_ID = "mod-gcp-white-soi-dev-1"
DATASET_ID = "mantak_database"
TABLE_ID = "classified_predictions_third_eye"
FULL_TABLE_PATH = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
SPLIT_GROUP_TABLE = f"{PROJECT_ID}.{DATASET_ID}.split_groups_third_eye"
SCENARIO_DETAILS = f"{PROJECT_ID}.{DATASET_ID}.scenario_details_third_eye"
ANNOTATIONS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.annotations_third_eye"

LOCAL_ANN_CSV = 'annotations_third_eye.csv'

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

def sync_local_data():
    """Automatically downloads annotations from BQ if the CSV is missing."""
    if not os.path.exists(LOCAL_ANN_CSV):
        with st.status("📥 Downloading dataset metadata for the first time..."):
            client = get_bigquery_client()
            query = f"SELECT * FROM `{ANNOTATIONS_TABLE}`"
            df = client.query(query).to_dataframe()
            df.to_csv(LOCAL_ANN_CSV, index=False)
            st.write(f"✅ Created {LOCAL_ANN_CSV}")

@st.cache_data
def get_filter_options(_client, column_name, table_path):
    query = f"SELECT DISTINCT CAST({column_name} AS STRING) as col FROM `{table_path}` WHERE {column_name} IS NOT NULL"
    return sorted(_client.query(query).to_dataframe()['col'].tolist())

@st.cache_data(ttl=3600)
def fetch_model_data(_client, selected_models, split, version, datasets, lighting, camera):
    if not selected_models: return pd.DataFrame()
    models_str = "', '".join(selected_models)
    where_clauses = [f"t1.model IN ('{models_str}')"]
    if split != "All": where_clauses.append(f"CAST(t2.split AS STRING) = '{split}'")
    if version != "All": where_clauses.append(f"CAST(t2.version AS STRING) = '{version}'")
    if datasets and "All" not in datasets:
        where_clauses.append(f"CAST(t3.dataset AS STRING) IN ('" + "','".join(datasets) + "')")
    if lighting and "All" not in lighting:
        where_clauses.append(f"CAST(t3.Lighting_Condition AS STRING) IN ('" + "','".join(lighting) + "')")
    if camera and "All" not in camera:
        where_clauses.append(f"CAST(t3.Camera_Movement AS STRING) IN ('" + "','".join(camera) + "')")

    query = f"""SELECT t1.frame, t1.confidence, t1.eval_type, t1.gt_id, t1.model as model_name, t1.scenario,
                       t1.xmin, t1.ymin, t1.xmax, t1.ymax, t1.class
                FROM `{FULL_TABLE_PATH}` AS t1
                LEFT JOIN `{SPLIT_GROUP_TABLE}` AS t2 ON t1.scenario = t2.scenario
                LEFT JOIN `{SCENARIO_DETAILS}` AS t3 ON t1.scenario = t3.scenario
                WHERE {" AND ".join(where_clauses)}"""
    df = _client.query(query).to_dataframe()
    if not df.empty:
        df['gt_key'] = df['scenario'].astype(str) + "_" + df['frame'].astype(str) + "_" + df['gt_id'].astype(str)
    return df


@st.cache_data
def load_annotations():
    # Ensure file exists before loading
    sync_local_data()
    if os.path.exists(LOCAL_ANN_CSV):
        df = pd.read_csv(LOCAL_ANN_CSV)
        df['gt_key'] = df['scenario'].astype(str) + "_" + df['frame'].astype(str) + "_" + df['id'].astype(str)
        cols = ['Occlusion_Level', 'Appearance_Level', 'Distance', 'Motion', 'Uniform']
        for col in cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)
        return df
    return pd.DataFrame()


# ============================================================================
# CALCULATION LOGIC
# ============================================================================

def define_unique_fp_detections(classified_pred_df, scenario_details_df, iou_threshold=0.8, return_clusters=False):
    fp_df = classified_pred_df[classified_pred_df['eval_type'] == 'FP'].copy()

    if fp_df.empty:
        if return_clusters:
            return 0, fp_df, {}
        else:
            return 0

    # Add a unique index to track detections
    fp_df = fp_df.reset_index(drop=True)
    fp_df['detection_idx'] = fp_df.index

    scenario_cluster = scenario_details_df[['scenario', 'cluster']].drop_duplicates()
    fp_df = fp_df.merge(scenario_cluster, on='scenario', how='left')

    group_by_cols = ['cluster', 'class']

    parent = {i: i for i in range(len(fp_df))}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    def vectorized_iou(boxes1, boxes2):
        # Expand dimensions for broadcasting
        boxes1_exp = boxes1[:, np.newaxis, :]  # (n, 1, 4)
        boxes2_exp = boxes2[np.newaxis, :, :]  # (1, n, 4)

        # Calculate intersection
        xa1 = np.maximum(boxes1_exp[:, :, 0], boxes2_exp[:, :, 0])
        ya1 = np.maximum(boxes1_exp[:, :, 1], boxes2_exp[:, :, 1])
        xa2 = np.minimum(boxes1_exp[:, :, 2], boxes2_exp[:, :, 2])
        ya2 = np.minimum(boxes1_exp[:, :, 3], boxes2_exp[:, :, 3])

        inter_w = np.maximum(0.0, xa2 - xa1)
        inter_h = np.maximum(0.0, ya2 - ya1)
        inter = inter_w * inter_h

        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Broadcast areas for union calculation
        area1_exp = area1[:, np.newaxis]  # (n, 1)
        area2_exp = area2[np.newaxis, :]  # (1, n)

        union = area1_exp + area2_exp - inter

        # Avoid division by zero
        iou_matrix = np.where(union > 0, inter / union, 0.0)

        return iou_matrix

    # Group by cluster and class (NOT frame - we want to track across frames)
    grouped = fp_df.groupby(group_by_cols)

    for group_key, group in grouped:
        n = len(group)

        if n < 2:
            continue  # No pairs to compare

        # Extract indices and boxes as numpy arrays (much faster than to_dict)
        indices = group['detection_idx'].values
        boxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values

        # Vectorized IoU calculation for all pairs at once
        iou_matrix = vectorized_iou(boxes, boxes)

        # Find pairs with IoU >= threshold
        # Use upper triangle to avoid duplicate comparisons
        i_indices, j_indices = np.where(np.triu(iou_matrix >= iou_threshold, k=1))

        # Union the overlapping detections
        for i, j in zip(i_indices, j_indices):
            union(indices[i], indices[j])

    # Find cluster representatives (one per unique FP)
    clusters = {}
    for idx in range(len(fp_df)):
        root = find(idx)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(idx)

    # Select one representative per cluster (the first one)
    unique_detection_indices = list(clusters.keys())

    # Count unique FPs
    unique_fp_count = len(unique_detection_indices)

    if return_clusters:
        return unique_fp_count, fp_df, clusters
    else:
        return unique_fp_count


def calculate_advanced_metrics(model_df, valid_gt_keys, total_gt_frames, d_h, fpiou_h, amount_of_frames):
    processed_df = model_df.copy()
    processed_df['effective_type'] = processed_df['eval_type'].replace({'D': d_h, 'FP_IOU': fpiou_h})
    processed_df = processed_df[processed_df['effective_type'] != 'Ignore']
    results = []
    is_fp = processed_df['effective_type'] == 'FP'
    is_tp_valid = (processed_df['effective_type'] == 'TP') & (processed_df['gt_key'].isin(valid_gt_keys))
    eval_subset = processed_df[is_fp | is_tp_valid]
    for t in np.arange(0.05, 1.0, 0.05):
        subset = eval_subset[eval_subset['confidence'] >= t]
        tp_sub = subset[subset['effective_type'] == 'TP']
        fp_count = len(subset[subset['effective_type'] == 'FP'])
        far = (fp_count / amount_of_frames * 300) if amount_of_frames > 0 else 0
        u_tp = tp_sub['gt_key'].nunique()
        rec = u_tp / total_gt_frames if total_gt_frames > 0 else 0
        prec = len(tp_sub) / (len(tp_sub) + fp_count) if (len(tp_sub) + fp_count) > 0 else 0
        results.append({'threshold': round(t, 2), 'Detection Recall': rec, 'False Alarm Rate': far,
                        'Detection Precision': prec,
                        'Detection f1': 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0,
                        'TP': len(tp_sub), 'FP': fp_count, 'FN': int(total_gt_frames - u_tp)})
    return pd.DataFrame(results)


def calculate_track_metrics(tp_sub, fp_sub, filtered_ann, total_clusters, scenario_details_df):
    gt_appearances = filtered_ann.groupby(['scenario', 'id'])['frame'].nunique().reset_index()
    gt_appearances.columns = ['scenario', 'gt_id', 'appearances']
    tp_per_gt = tp_sub.groupby(['scenario', 'gt_id']).size().reset_index(name='tp_count')
    track_df = gt_appearances.merge(tp_per_gt, on=['scenario', 'gt_id'], how='left').fillna(0)
    track_df['is_successful'] = (track_df['tp_count'] / track_df['appearances']) >= 0.5
    track_df['is_alert'] = track_df['tp_count'] > 0
    total_tracks = len(track_df)
    track_rec_obs = track_df['is_successful'].sum() / total_tracks if total_tracks > 0 else 0
    track_rec_alrt = track_df['is_alert'].sum() / total_tracks if total_tracks > 0 else 0

    unique_fp_count = define_unique_fp_detections(fp_sub, scenario_details_df)

    track_far = round(unique_fp_count / total_clusters, 2) if total_clusters > 0 else 0
    return {'Track Recall Obs': track_rec_obs, 'Track Recall Alerts': track_rec_alrt, 'Track FAR': track_far,
            'Amount of Track FP': unique_fp_count,
            'Avg TPs Confidence': tp_sub['confidence'].mean() if not tp_sub.empty else 0}


def color_metrics(val, target='high'):
    if target == 'high':
        color = '#c7e9c0' if val > 0.8 else '#ffcdd2' if val < 0.3 else 'transparent'
    else:
        color = '#c7e9c0' if val < 2.0 else '#ffcdd2' if val > 10.0 else 'transparent'
    return f'background-color: {color}'


# ============================================================================
# APP UI
# ============================================================================
client = get_bigquery_client()
ann_df = load_annotations()

st.sidebar.header("General Settings")
all_models_list = get_filter_options(client, "model", FULL_TABLE_PATH)
selected_models = st.sidebar.multiselect("Select Models", all_models_list, max_selections=5)

for m in selected_models:
    if m not in st.session_state.selected_thresholds: st.session_state.selected_thresholds[m] = 0.5

st.sidebar.subheader("Scenario Filters")
s_split = st.sidebar.selectbox("Split", ["All"] + get_filter_options(client, "split", SPLIT_GROUP_TABLE))
s_ver = st.sidebar.selectbox("Version", ["All"] + get_filter_options(client, "version", SPLIT_GROUP_TABLE))
s_ds = st.sidebar.multiselect("DataSets", ["All"] + get_filter_options(client, "dataset", SCENARIO_DETAILS),
                              default=["All"])
s_light = st.sidebar.multiselect("Lighting",
                                 ["All"] + get_filter_options(client, "Lighting_Condition", SCENARIO_DETAILS),
                                 default=["All"])
s_cam = st.sidebar.multiselect("Camera", ["All"] + get_filter_options(client, "Camera_Movement", SCENARIO_DETAILS),
                               default=["All"])

st.sidebar.subheader("Object Attribute Filters")


def attr_sel(label, col):
    opts = sorted(ann_df[col].unique().tolist()) if col in ann_df.columns else []
    sel = st.sidebar.multiselect(label, ["All"] + opts, default=["All"])
    return opts if "All" in sel or not sel else sel


f_app, f_occ, f_dist, f_mot, f_uni = [attr_sel(l, c) for l, c in [
    ("Appearance", "Appearance_Level"), ("Occlusion", "Occlusion_Level"),
    ("Distance", "Distance"), ("Motion", "Motion"), ("Uniform", "Uniform")]]
d_h = st.sidebar.selectbox("Handle 'D' as:", ["Ignore", "TP", "FP"])
fpiou_h = st.sidebar.selectbox("Handle 'FP_IOU' as:", ["Ignore", "FP", "TP"], index=2)

if selected_models and not ann_df.empty:
    with st.spinner("Processing..."):
        full_df = fetch_model_data(client, selected_models, s_split, s_ver, s_ds, s_light, s_cam)
        meta_where = []
        if s_split != "All": meta_where.append(f"CAST(t2.split AS STRING) = '{s_split}'")
        if s_ver != "All": meta_where.append(f"CAST(t2.version AS STRING) = '{s_ver}'")
        if s_ds and "All" not in s_ds: meta_where.append(f"CAST(t1.dataset AS STRING) IN ('" + "','".join(s_ds) + "')")
        if s_light and "All" not in s_light: meta_where.append(
            f"CAST(t1.Lighting_Condition AS STRING) IN ('" + "','".join(s_light) + "')")
        if s_cam and "All" not in s_cam: meta_where.append(
            f"CAST(t1.Camera_Movement AS STRING) IN ('" + "','".join(s_cam) + "')")

        where_str = " WHERE " + " AND ".join(meta_where) if meta_where else ""
        metadata_query = f"SELECT COUNT(DISTINCT t1.scenario) as num_sc, SUM(t1.amount_of_frames) as tot_fr, COUNT(DISTINCT t1.cluster) as total_clusters, ARRAY_AGG(DISTINCT t1.scenario) as sc_list FROM `{SCENARIO_DETAILS}` AS t1 LEFT JOIN `{SPLIT_GROUP_TABLE}` AS t2 ON t1.scenario = t2.scenario {where_str}"
        m_res = client.query(metadata_query).to_dataframe().iloc[0]

        num_scenarios = int(m_res['num_sc']) if not pd.isna(m_res['num_sc']) else 0
        amount_of_frames = int(m_res['tot_fr']) if not pd.isna(m_res['tot_fr']) else 1
        total_clusters = int(m_res['total_clusters']) if not pd.isna(m_res['total_clusters']) else 0
        relevant_sc_list = list(m_res['sc_list']) if not pd.isna(m_res['sc_list']).all() else []

        filtered_ann = ann_df[(ann_df['scenario'].isin(relevant_sc_list)) & (ann_df['Appearance_Level'].isin(f_app)) & (
            ann_df['Occlusion_Level'].isin(f_occ)) & (ann_df['Distance'].isin(f_dist)) & (
                                  ann_df['Motion'].isin(f_mot)) & (ann_df['Uniform'].isin(f_uni))]
        total_gt = filtered_ann['gt_key'].nunique()
        valid_keys = set(filtered_ann['gt_key'].tolist())
        total_gt_tracks = filtered_ann[['scenario', 'id']].drop_duplicates().shape[0]

        # RESTORED 5-COLUMN OVERVIEW
        st.markdown("### 📊 Dataset Overview")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Number of Scenarios", f"{num_scenarios:,}")
        m_col2.metric("Total Frames", f"{amount_of_frames:,}")
        m_col3.metric("Total Clusters", f"{total_clusters:,}")
        m_col4.metric("Total GT Tracks", f"{total_gt_tracks:,}")
        m_col5.metric("Relevant Objects", f"{total_gt:,}")
        st.markdown("---")

        if not full_df.empty:
            st.subheader("📈 Interactive ROC Curve")
            roc_fig = go.Figure()
            all_metrics, trace_map, trace_counter = {}, {}, 0
            for i, m in enumerate(selected_models):
                res = calculate_advanced_metrics(full_df[full_df['model_name'] == m], valid_keys, total_gt, d_h,
                                                 fpiou_h, amount_of_frames)
                all_metrics[m], color = res, MY_COLORS[i % len(MY_COLORS)]
                roc_fig.add_trace(
                    go.Scatter(x=res['False Alarm Rate'], y=res['Detection Recall'], name=m, mode='lines+markers',
                               line=dict(color=color, width=3), customdata=res['threshold'],
                               hovertemplate="<b>" + m + "</b><br>Th: %{customdata:.2f}<br>Recall: %{y:.2%}<br>FAR: %{x:.2f}<extra></extra>"))
                trace_map[trace_counter] = m;
                trace_counter += 1
                curr_t = st.session_state.selected_thresholds.get(m, 0.5);
                sel_row = res.iloc[(res['threshold'] - curr_t).abs().idxmin()]
                roc_fig.add_trace(
                    go.Scatter(x=[sel_row['False Alarm Rate']], y=[sel_row['Detection Recall']], mode='markers',
                               marker=dict(color=color, size=15, line=dict(color='white', width=2)), showlegend=False,
                               hoverinfo='skip'))
                trace_counter += 1
            roc_fig.update_layout(xaxis_title="False Alarm Rate (per 300 Frames)", yaxis_title="Recall",
                                  template='plotly_white', height=500, clickmode='event+select', hovermode='closest')
            event = st.plotly_chart(roc_fig, use_container_width=True, on_select="rerun")
            if event and "selection" in event and len(event["selection"]["points"]) > 0:
                p = event["selection"]["points"][0];
                m_name = trace_map.get(p["curve_number"]) or trace_map.get(p["curve_number"] - 1)
                if m_name: st.session_state.selected_thresholds[m_name] = p["customdata"]; st.rerun()

            st.subheader("📋 Comparison Summary")
            comp_list = []
            sc_details_df = client.query(f"SELECT scenario, cluster FROM `{SCENARIO_DETAILS}`").to_dataframe()
            for m in selected_models:
                t = st.session_state.selected_thresholds.get(m, 0.5)
                m_df = full_df[full_df['model_name'] == m]
                row = all_metrics[m].iloc[(all_metrics[m]['threshold'] - t).abs().idxmin()].to_dict()
                subset = m_df[m_df['confidence'] >= t].copy()
                subset['effective_type'] = subset['eval_type'].replace({'D': d_h, 'FP_IOU': fpiou_h})
                tp_sub = subset[(subset['effective_type'] == 'TP') & (subset['gt_key'].isin(valid_keys))]
                fp_sub = subset[subset['effective_type'] == 'FP']
                track_stats = calculate_track_metrics(tp_sub, fp_sub, filtered_ann, total_clusters, sc_details_df)
                row.update(track_stats);
                row['Model'] = m;
                comp_list.append(row)

            comp_df = pd.DataFrame(comp_list)
            best_m = comp_df.loc[comp_df['Detection f1'].idxmax(), 'Model']
            st.dataframe(comp_df[
                             ['Model', 'Detection Recall', 'False Alarm Rate', 'Detection Precision', 'Detection f1',
                              'Track Recall Obs', 'Track Recall Alerts', 'Track FAR', 'Amount of Track FP',
                              'Avg TPs Confidence', 'TP', 'FP', 'FN']].style.apply(
                lambda r: ['color: #FF1493; font-weight: bold' if r['Model'] == best_m else '' for _ in r], axis=1).map(
                lambda x: color_metrics(x, 'high'),
                subset=['Detection Recall', 'Detection Precision', 'Detection f1']).map(
                lambda x: color_metrics(x, 'low'), subset=['False Alarm Rate']).format(
                {'Detection Recall': '{:.2%}', 'Detection Precision': '{:.2%}', 'Detection f1': '{:.2%}',
                 'False Alarm Rate': '{:.2f}', 'Track Recall Obs': '{:.2%}', 'Track Recall Alerts': '{:.2%}',
                 'Avg TPs Confidence': '{:.2f}', 'Track FAR': '{:.2f}', 'Amount of Track FP': '{:,.0f}', 'TP': '{:.0f}',
                 'FP': '{:.0f}', 'FN': '{:.0f}'}), use_container_width=True)

            st.markdown("---")
            card_cols = st.columns(len(selected_models))
            for i, m in enumerate(selected_models):
                data = next(item for item in comp_list if item['Model'] == m)
                curr_t = st.session_state.selected_thresholds.get(m, 0.5)
                bg, border = ("#fff0f6", "#FF1493") if m == best_m else ("#ffffff", "#ddd")
                with card_cols[i]:
                    st.markdown(
                        f"""<div style="padding:15px; border-radius:10px; border: 1px solid {border}; background-color: {bg}; text-align:center; position: relative;">
                        <div style="position: absolute; top: 10px; right: 10px; font-size: 0.8em; font-weight: bold;">Conf: {curr_t:.2f}</div>
                        <h4 style="margin-top: 20px;">{m}</h4>
                        <p style="font-size:1.6em; font-weight:bold; color:#FF1493; margin: 10px 0;">{data['Detection f1']:.1%}</p>
                        <p style="font-size:0.85em; color: #555;">FAR: {data['False Alarm Rate']:.2f} | Track FP: {int(data['Amount of Track FP'])}</p>
                    </div>""", unsafe_allow_html=True)
else:
    st.info("Please select models to begin.")

