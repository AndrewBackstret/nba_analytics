import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import glob
import time
from typing import List, Optional
try:
    import pyarrow.dataset as pa_ds 
except Exception:
    pa_ds = None


st.set_page_config(
    page_title="NBA Analytics Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏀 NBA Analytics Dashboard")
st.markdown("---")

@st.cache_data
def load_data() -> Optional[pd.DataFrame]:

    candidate_dirs: List[str] = [
        "/opt/spark-data/gold",
        "data/gold",
        "../data/gold",
    ]

    def has_parquet_files(base_dir: str) -> bool:
        return (
            os.path.isdir(base_dir)
            and (
                glob.glob(os.path.join(base_dir, "*.parquet"))
                or glob.glob(os.path.join(base_dir, "*.snappy.parquet"))
                or glob.glob(os.path.join(base_dir, "**", "*.parquet"), recursive=True)
                or glob.glob(os.path.join(base_dir, "**", "*.snappy.parquet"), recursive=True)
            )
        )

    try:
        target_dir = None
        for d in candidate_dirs:
            if has_parquet_files(d):
                target_dir = d
                break

        if target_dir is None:
            st.error("❌ No Parquet files found under gold/silver paths.")
            return None

        if pa_ds is not None:
            dataset = pa_ds.dataset(target_dir, format="parquet")
            table = dataset.to_table()
            df = table.to_pandas()
            return df

        file_paths = []
        file_paths += glob.glob(os.path.join(target_dir, "*.parquet"))
        file_paths += glob.glob(os.path.join(target_dir, "*.snappy.parquet"))
        file_paths += glob.glob(os.path.join(target_dir, "**", "*.parquet"), recursive=True)
        file_paths += glob.glob(os.path.join(target_dir, "**", "*.snappy.parquet"), recursive=True)
        file_paths = sorted(set(file_paths))

        if not file_paths:
            st.error(f"❌ No Parquet files inside: {target_dir}")
            return None

        frames: List[pd.DataFrame] = []
        for p in file_paths:
            try:
                frames.append(pd.read_parquet(p))
            except Exception as e:
                st.warning(f"Could not read {p}: {e}")
        if not frames:
            st.error("❌ Could not read any valid Parquet file.")
            return None

        df = pd.concat(frames, ignore_index=True)
        return df

    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")
        return None

df = load_data()

if df is not None and not st.session_state.get("shown_data_loaded", False):
    st.toast("✅ Data loaded successfully")
    st.session_state["shown_data_loaded"] = True

if df is not None:
    for col in [
        'points', 'rebounds', 'assists', 'games_played',
        'field_goal_percentage', 'three_point_field_goal_percentage',
        'free_throw_percentage', 'steals', 'blocks', 'turnovers',
        'offensive_rebounds', 'defensive_rebounds'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

if df is not None:
    st.sidebar.header("🔍 Filters")
    
    players = ['All'] + sorted(df['player_name'].unique().tolist())
    selected_player = st.sidebar.selectbox(
        "👤 Select player:",
        players,
        index=0
    )
    
    filtered_df = df.copy()
    
    if selected_player != 'All':
        filtered_df = filtered_df[filtered_df['player_name'] == selected_player]
    
    
    with st.sidebar.expander("📊 Metric filters", expanded=False):

        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        

        main_metrics = ['games_played', 'points', 'rebounds', 'assists', 'field_goal_percentage', 'three_point_field_goal_percentage']
        
        metric_filters = {}
        for metric in main_metrics:
            if metric in numeric_columns:
                min_val = float(df[metric].min())
                max_val = float(df[metric].max())
                metric_filters[metric] = st.slider(
                    f"{metric.replace('_', ' ').title()}:",
                    min_val, max_val, (min_val, max_val)
                )
    

    for metric, (min_val, max_val) in metric_filters.items():
        filtered_df = filtered_df[
            (filtered_df[metric] >= min_val) & 
            (filtered_df[metric] <= max_val)
        ]

    st.sidebar.header("📈 Sorting")
    
    sort_column = st.sidebar.selectbox(
        "Sort by:",
        numeric_columns,
        index=numeric_columns.index('points') if 'points' in numeric_columns else 0
    )
    
    sort_order = st.sidebar.radio(
        "Order:",
        ["Descending", "Ascending"]
    )
    

    ascending = sort_order == "Ascending"
    filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

    filtered_df = filtered_df.copy()
    filtered_df.insert(0, 'Rank', range(1, len(filtered_df) + 1))
    
    
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] button {font-size: 1.05rem !important; padding: 0.6rem 1rem !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Table", "📈 Visualizations", "🏆 Top Performers", "🤝 Comparisons"])
    
    with tab1:
        st.subheader("Filtered Data")
        
        display_columns = st.multiselect(
            "Select columns to display:",
            df.columns.tolist(),
            default=['player_name', 'points', 'rebounds', 'assists', 'field_goal_percentage', 'games_played']
        )
        
        if display_columns:

            cols_to_show = ['Rank'] + [c for c in display_columns if c != 'Rank']
            st.dataframe(
                filtered_df[cols_to_show],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Please select at least one column to display.")
    
    with tab2:
        st.subheader("Visualizations")

        # Bar Chart
        if len(filtered_df) > 0:
            top_10 = filtered_df.head(10)
            fig_bar = px.bar(
                top_10,
                x='player_name',
                y=sort_column,
                title=f"Top 10 players by {sort_column.replace('_', ' ').title()}"
            )
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No data available for the chart.")

        # Bubble Chart
        st.markdown("---")
        with st.expander("Bubble chart (advanced)", expanded=False):
            if len(filtered_df) > 0:
                c1, c2 = st.columns([1, 2])
                with c1:

                    st.markdown("<div style='display:flex;flex-direction:column;justify-content:center;height:100%'>", unsafe_allow_html=True)
                    local_numeric = [c for c in numeric_columns if c != 'player_name']
                    lx = st.selectbox("X axis", local_numeric, index=local_numeric.index('points') if 'points' in local_numeric else 0, key='viz_x')
                    ly = st.selectbox("Y axis", local_numeric, index=local_numeric.index('rebounds') if 'rebounds' in local_numeric else 0, key='viz_y')
                    lcolor = st.selectbox("Color by", local_numeric, index=local_numeric.index('assists') if 'assists' in local_numeric else 0, key='viz_color')
                    lsize = st.selectbox("Bubble size", local_numeric, index=local_numeric.index('games_played') if 'games_played' in local_numeric else 0, key='viz_size')
                    st.markdown("</div>", unsafe_allow_html=True)
                with c2:
                    fig_scatter = px.scatter(
                        filtered_df,
                        x=lx,
                        y=ly,
                        color=lcolor,
                        size=lsize,
                        hover_data=['player_name'],
                        title=f"{lx.title()} vs {ly.title()} (size = {lsize.replace('_',' ')}, color = {lcolor.replace('_',' ')})"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No data available for the bubble chart.")

        # Correlation matrix inside expander
        if len(filtered_df) > 1:
            with st.expander("Correlation Matrix", expanded=False):
                correlation_cols = [c for c in ['points', 'rebounds', 'assists', 'field_goal_percentage', 'three_point_field_goal_percentage'] if c in filtered_df.columns]
                if len(correlation_cols) >= 2:
                    corr_data = filtered_df[correlation_cols].corr()
                    fig_corr = px.imshow(
                        corr_data,
                        text_auto=True,
                        aspect="auto",
                        title="Main Metrics Correlation"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Not enough numeric columns to compute correlations.")
    
    with tab3:
        st.subheader("🏆 Top Performers")
        
        top_metrics = ['points', 'rebounds', 'assists', 'field_goal_percentage', 'steals', 'blocks']
        
        for i, metric in enumerate(top_metrics):
            if metric in df.columns and len(filtered_df) > 0:
                col = st.columns(2)[i % 2]
                with col:
                    top_player = filtered_df.nlargest(1, metric)
                    if len(top_player) > 0:
                        player_name = top_player.iloc[0]['player_name']
                        value = top_player.iloc[0][metric]
                        st.metric(
                            f"🥇 {metric.replace('_', ' ').title()}",
                            f"{player_name}",
                            f"{value:.2f}"
                        )
    
    with tab4:
        st.subheader("Compare players")
        if len(df) > 0:
            players_only = sorted(df['player_name'].dropna().unique().tolist())
            c1, c2 = st.columns(2)
            with c1:
                player_a = st.selectbox("First player", players_only, key="cmp_player_a")
            with c2:
                player_b = st.selectbox("Second player", players_only, index=min(1, len(players_only)-1), key="cmp_player_b")

            default_metrics = ['points', 'rebounds', 'assists']
            available_metrics = [m for m in ['games_played', 'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'field_goal_percentage', 'three_point_field_goal_percentage', 'free_throw_percentage'] if m in df.columns]
            metrics_to_compare = st.multiselect("Metrics to compare", available_metrics, default=[m for m in default_metrics if m in available_metrics])

            if player_a == player_b:
                st.info("Please select two different players.")
            elif not metrics_to_compare:
                st.info("Select at least one metric to compare.")
            else:
                cmp_df = df[df['player_name'].isin([player_a, player_b])].copy()

                cmp_df = cmp_df.groupby('player_name').tail(1)
                plot_df = cmp_df[['player_name'] + metrics_to_compare].melt(id_vars='player_name', var_name='Metric', value_name='Value')
                fig_cmp = px.bar(
                    plot_df,
                    x='Metric', y='Value', color='player_name', barmode='group',
                    title=f"Comparison: {player_a} vs {player_b}"
                )

                fig_cmp.update_traces(marker_color=None)
                warm_palette = ['#E67E22', '#D35400', '#F39C12', '#BA4A00']
                fig_cmp.update_layout(colorway=warm_palette, legend_title_text="Player")
                fig_cmp.update_xaxes(tickangle=45)
                st.plotly_chart(fig_cmp, use_container_width=True)

else:
    st.error("Data could not be loaded. Please verify Parquet files are available.")
    

    st.subheader("🔍 Debug info")
    st.write("Checked folders:")
    candidate_dirs = [
        "/opt/spark-data/gold",
        "data/gold",
        "../data/gold",
    ]
    for d in candidate_dirs:
        exists = os.path.isdir(d)
        count = len(glob.glob(os.path.join(d, '**', '*.parquet'), recursive=True)) if exists else 0
        st.write(f"- {d}: {'✅ Exists' if exists else '❌ Missing'} | parquet files: {count}")
    
    st.write(f"Actual Directory: {os.getcwd()}")
    st.write(f"Files in Actual Directory: {os.listdir('.')}")
