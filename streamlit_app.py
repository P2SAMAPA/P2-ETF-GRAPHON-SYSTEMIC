"""
streamlit_app.py  —  Graphon ETF Systemic Risk Dashboard
"""

import streamlit as st
import pandas as pd
import requests
import json
import glob
from datetime import datetime

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
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .confidence-high { color: #27ae60; font-weight: 600; }
    .confidence-medium { color: #f39c12; font-weight: 600; }
    .confidence-low { color: #e74c3c; font-weight: 600; }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #7f8c8d;
    }
    .phase-tag {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .phase-stable { background: #27ae60; color: white; }
    .phase-transition { background: #f39c12; color: white; }
    .phase-breakdown { background: #e74c3c; color: white; }
    </style>
""", unsafe_allow_html=True)


def load_data():
    """Load latest results."""
    # Try local first
    json_files = glob.glob("graphon_results_*.json")
    if json_files:
        latest = sorted(json_files)[-1]
        with open(latest, 'r') as f:
            return json.load(f)
    
    # Try HuggingFace
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
        st.info("Check that the file exists: graphon_results_*.json")
        return
    
    run_date = data.get('run_date', 'Unknown')
    st.caption(f"Results from: {run_date}")
    
    # Check if data has the required keys
    top_picks = data.get('top_picks', {})
    phase_transitions = data.get('phase_transitions', {})
    best_windows = data.get('best_windows', {})
    graphon_metrics = data.get('graphon_metrics', {})
    
    if not top_picks:
        st.warning("No top picks found in the data")
        st.json(data)  # Show raw data for debugging
        return
    
    tab1, tab2 = st.tabs(["📊 Top Picks", "🌐 Network Phase"])
    
    with tab1:
        st.subheader("Top ETF Picks by Universe")
        
        for universe, picks in top_picks.items():
            phase = phase_transitions.get(universe, {})
            phase_type = phase.get('phase_type', 'stable')
            best_win = best_windows.get(universe, {}).get('window', 252)
            
            # Phase tag
            if phase_type == 'stable':
                phase_class = 'phase-stable'
                phase_label = 'Stable'
            elif phase_type in ['connectivity_buildup', 'core_periphery_formation']:
                phase_class = 'phase-transition'
                phase_label = 'Transitioning'
            else:
                phase_class = 'phase-breakdown'
                phase_label = 'Breaking Down'
            
            st.markdown(f"""
            ### {universe}
            **Best Window:** {best_win} days | **Phase:** <span class="phase-tag {phase_class}">{phase_label}</span>
            """, unsafe_allow_html=True)
            
            if not picks:
                st.info("No picks available")
                st.markdown("---")
                continue
            
            cols = st.columns(min(len(picks), 3))
            for i, pick in enumerate(picks):
                with cols[i % len(cols)]:
                    conf = pick.get('confidence', 'low').lower()
                    if conf not in ['high', 'medium', 'low']:
                        conf = 'low'
                    
                    st.markdown(f"""
                    <div class="ticker-card">
                        <h3 style="margin:0; font-size:1.3rem;">{pick.get('ticker', 'N/A')}{' ⭐' if pick.get('is_core', False) else ''}</h3>
                        <div style="font-size:2rem; font-weight:700; margin:0.3rem 0;">
                            {pick.get('expected_return', 0):.1f}%
                        </div>
                        <div class="confidence-{conf}">Confidence: {pick.get('confidence', 'Low')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    with tab2:
        st.subheader("Graphon Phase Analysis")
        
        for universe in top_picks.keys():
            phase = phase_transitions.get(universe, {})
            metrics = graphon_metrics.get(universe, {})
            
            st.markdown(f"### {universe}")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Phase", phase.get('phase_type', 'stable').upper())
            with col2:
                st.metric("Connectivity", f"{metrics.get('integrated_connectivity', 0):.3f}")
            with col3:
                st.metric("Entropy", f"{metrics.get('entropy', 0):.3f}")
            with col4:
                st.metric("Core Ratio", f"{metrics.get('core_ratio', 0):.2f}")
            
            # Phase interpretation
            phase_type = phase.get('phase_type', 'stable')
            transition_detected = phase.get('transition_detected', False)
            
            if transition_detected:
                if phase_type == 'connectivity_breakdown':
                    st.warning("⚠️ **Connectivity Breakdown** - Network fragmenting. Systemic risk increasing.")
                elif phase_type == 'core_periphery_formation':
                    st.warning("⚠️ **Core-Periphery Formation** - Network concentrating. Watch for crowded trades.")
                elif phase_type == 'connectivity_buildup':
                    st.info("🔄 **Connectivity Buildup** - Network becoming more connected. Potential contagion risk.")
                else:
                    st.info("🔄 **Transition Detected** - Network phase shifting.")
            else:
                st.success("✅ **Stable** - No significant phase transition detected.")
            
            # Core ETFs
            core_etfs = phase.get('core_etfs', [])
            if core_etfs:
                st.markdown(f"**Core ETFs (80th percentile):** {', '.join(core_etfs[:8])}")
            else:
                st.caption("No core ETFs identified")
            
            st.markdown("---")


if __name__ == "__main__":
    main()
