"""
trainer.py  —  Graphon ETF Systemic Risk Trainer
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from graphon_model import GraphonETFModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_trainer() -> Dict:
    """Main trainer."""
    
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}
    
    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date,
        "top_picks": {},
        "graphon_metrics": {},
        "phase_transitions": {},
        "best_windows": {},
        "universes": {}
    }
    
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {universe_name}...")
        
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue
        
        prices = prices_df[available].dropna().values
        returns = np.diff(np.log(prices), axis=0)
        
        # Align macro
        macro = macro_df.values
        if len(macro) > len(returns):
            macro = macro[-len(returns):]
        elif len(macro) < len(returns):
            pad = len(returns) - len(macro)
            macro = np.vstack([macro[:1]] * pad + [macro])
        
        if len(returns) < 50:
            logger.warning(f"Not enough data for {universe_name}")
            continue
        
        # Test windows
        window_results = {}
        best_score = -999
        best_window = 252
        
        for window in config.WINDOWS:
            if len(returns) < window + 50:
                continue
            
            # Reset model for each window
            model = GraphonETFModel(config.GRAPHON_CONFIG)
            returns_window = returns[-window:]
            macro_window = macro[-window:]
            
            result = model.analyze_universe(returns_window, macro_window, available)
            
            if "error" not in result:
                metrics = result.get("metrics", {})
                phase = result.get("phase", {})
                
                window_results[window] = {
                    "metrics": metrics,
                    "phase": phase,
                    "picks": result.get("picks", []),
                    "core_etfs": result.get("core_etfs", [])
                }
                
                # Score based on connectivity and phase detection
                score = metrics.get("integrated_connectivity", 0)
                if phase.get("transition_detected", False):
                    score += 0.3
                
                if score > best_score:
                    best_score = score
                    best_window = window
        
        # Use best window
        if best_window in window_results:
            final = window_results[best_window]
            picks = final.get("picks", [])
            metrics = final.get("metrics", {})
            phase = final.get("phase", {})
            core_etfs = final.get("core_etfs", [])
        else:
            # Fallback
            model = GraphonETFModel(config.GRAPHON_CONFIG)
            fallback = model.analyze_universe(returns[-252:], macro[-252:], available)
            picks = fallback.get("picks", [])
            metrics = fallback.get("metrics", {})
            phase = fallback.get("phase", {})
            core_etfs = fallback.get("core_etfs", [])
        
        results["top_picks"][universe_name] = picks
        results["graphon_metrics"][universe_name] = metrics
        results["phase_transitions"][universe_name] = {
            "phase_type": phase.get("phase_type", "stable"),
            "transition_detected": phase.get("transition_detected", False),
            "confidence": phase.get("confidence", 0),
            "core_etfs": core_etfs[:10]
        }
        results["best_windows"][universe_name] = {"window": best_window}
        results["universes"][universe_name] = {
            "tickers": available,
            "window_results": window_results
        }
        
        logger.info(f"  ✅ Best window: {best_window}")
        logger.info(f"  ✅ Phase: {phase.get('phase_type', 'stable')}")
        logger.info(f"  ✅ Core ETFs: {core_etfs[:5]}")
        logger.info(f"  ✅ Top picks:")
        for pick in picks:
            logger.info(f"     {pick['ticker']}: {pick['expected_return']}% ({pick['confidence']})")
    
    # Save
    output_path = f"graphon_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Saved: {output_path}")
    
    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    
    return results


if __name__ == "__main__":
    run_trainer()
