"""
streamlit_app.py  —  Graphon ETF Systemic Risk Dashboard
"""

import streamlit as st
import pandas as pd
import requests
import json
import glob
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="P2-GRAPHON-SYSTEMIC",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .ticker-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
    }
    .confidence-high { color: #27ae60; font-weight: 600; }
    .confidence-medium { color: #f39c12; font-weight: 600; }
    .confidence-low { color: #e74c3c; font-weight: 600; }
    .phase-warning { color: #e74c3c; font-weight: 700; }
    .phase-stable { color: #27ae60; font-weight: 700; }
    .phase-transition { color: #f39c12; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)


def load_data():
    """Load latest results."""
    json_files = glob.glob("graphon_results_*.json")
    if json_files:
        latest = sorted(json_files)[-1]
        with open(latest, 'r') as f:
            return json.load(f)
    
    try:
        repo_id = "P2SAMAPA/p2-etf-graphon-systemic-results"
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/graphon_results_{today}.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    return None


def main():
    st.markdown('<div class="main-header">🌐 P2-GRAPHON-SYSTEMIC</div>', unsafe_allow_html=True)
    st.markdown("*Graphon-based ETF Network Analysis for Systemic Risk*")
    
    data = load_data()
    
    if not data:
        st.error("No data available. Run `python trainer.py` first.")
        return
    
    run_date = data.get('run_date', 'Unknown')
    st.caption(f"Results from: {run_date}")
    
    tab1, tab2 = st.tabs(["📊 Top Picks", "🌐 Network Phase"])
    
    top_picks = data.get('top_picks', {})
    phase_transitions = data.get('phase_transitions', {})
    best_windows = data.get('best_windows', {})
    graphon_metrics = data.get('graphon_metrics', {})
    
    with tab1:
        st.subheader("Top ETF Picks by Universe")
        
        for universe, picks in top_picks.items():
            phase = phase_transitions.get(universe, {})
            phase_type = phase.get('phase_type', 'stable')
            core_etfs = phase.get('core_etfs', [])
            best_win = best_windows.get(universe, {}).get('window', 252)
            
            st.markdown(f"### {universe}")
            st.markdown(f"**Best Window:** {best_win} days | **Phase:** {phase_type.upper()}")
            
            if core_etfs:
                st.caption(f"Core ETFs: {', '.join(core_etfs[:5])}")
            
            cols = st.columns(min(len(picks), 3))
            for i, pick in enumerate(picks):
                with cols[i % len(cols)]:
                    conf = pick['confidence'].lower()
                    color = "#27ae60" if conf == "high" else "#f39c12" if conf == "medium" else "#e74c3c"
                    is_core = pick.get('is_core', False)
                    
                    st.markdown(f"""
                    <div class="ticker-card">
                        <h3 style="margin:0;">{pick['ticker']}{' ⭐' if is_core else ''}</h3>
                        <div style="font-size:2rem; font-weight:700; margin:0.5rem 0;">
                            {pick['expected_return']:.1f}%
                        </div>
                        <div style="color:{color}; font-weight:600;">Confidence: {pick['confidence']}</div>
                        <div style="font-size:0.7rem; color:#888; margin-top:0.3rem;">
                            {'Core ETF' if is_core else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    with tab2:
        st.subheader("Graphon Phase Analysis")
        
        for universe in top_picks.keys():
            st.markdown(f"### {universe}")
            
            phase = phase_transitions.get(universe, {})
            metrics = graphon_metrics.get(universe, {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Phase Type", phase.get('phase_type', 'stable').upper())
            with col2:
                st.metric("Integrated Connectivity", f"{metrics.get('integrated_connectivity', 0):.3f}")
            with col3:
                st.metric("Entropy", f"{metrics.get('entropy', 0):.3f}")
            with col4:
                st.metric("Core Ratio", f"{metrics.get('core_ratio', 0):.2f}")
            
            # Phase interpretation
            phase_type = phase.get('phase_type', 'stable')
            if phase_type == 'connectivity_breakdown':
                st.warning("⚠️ **Connectivity Breakdown Detected** - Network is fragmenting. Increased systemic risk.")
            elif phase_type == 'core_periphery_formation':
                st.warning("⚠️ **Core-Periphery Formation** - Network concentrating. Watch for crowded trades.")
            elif phase_type == 'connectivity_buildup':
                st.info("🔄 **Connectivity Buildup** - Network becoming more connected. Potential for contagion.")
            else:
                st.success("✅ **Stable** - No significant phase transition detected.")
            
            # Core ETFs
            core_etfs = phase.get('core_etfs', [])
            if core_etfs:
                st.markdown(f"**Core ETFs (80th percentile connectivity):** {', '.join(core_etfs[:10])}")
            
            st.markdown("---")


if __name__ == "__main__":
    main()
