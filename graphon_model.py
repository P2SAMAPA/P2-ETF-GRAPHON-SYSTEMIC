"""
graphon_model.py  —  Graphon Model for ETF Networks
"""

import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GraphonETFModel:
    """
    Graphon model for ETF network analysis.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_bins = config.get("n_bins", 20)
        self.bandwidth = config.get("bandwidth", 0.1)
        self.transition_threshold = config.get("transition_threshold", 0.15)
        self.W_history = []
        self.metrics_history = []
        
    def build_adjacency_matrix(self, returns: np.ndarray, macro: np.ndarray) -> np.ndarray:
        """Build adjacency matrix W_t(i,j)."""
        n_samples, n_etfs = returns.shape
        
        # Correlation
        corr = np.corrcoef(returns.T)
        
        # Volatility covariance
        vol = np.std(returns, axis=0)
        vol_cov = np.outer(vol, vol) * np.corrcoef(returns.T)
        vol_cov_norm = vol_cov / (np.max(vol_cov) + 1e-8)
        
        # Combined
        W = 0.6 * np.abs(corr) + 0.4 * vol_cov_norm
        np.fill_diagonal(W, 0)
        
        return W
    
    def estimate_graphon(self, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate graphon from adjacency matrix."""
        n = W.shape[0]
        
        # Sort by degree
        degrees = W.sum(axis=1)
        sorted_indices = np.argsort(degrees)[::-1]
        W_sorted = W[sorted_indices][:, sorted_indices]
        
        # Bin
        bin_size = max(1, n // self.n_bins)
        n_bins_actual = min(self.n_bins, max(2, n // bin_size))
        
        graphon = np.zeros((n_bins_actual, n_bins_actual))
        node_positions = np.zeros(n)
        
        for i in range(n):
            node_positions[i] = min(i // bin_size, n_bins_actual - 1) / n_bins_actual
        
        for i in range(n_bins_actual):
            for j in range(n_bins_actual):
                i_start = i * bin_size
                i_end = min((i + 1) * bin_size, n)
                j_start = j * bin_size
                j_end = min((j + 1) * bin_size, n)
                
                if i_start < i_end and j_start < j_end:
                    block = W_sorted[i_start:i_end, j_start:j_end]
                    graphon[i, j] = np.mean(block) if block.size > 0 else 0
        
        # Smooth
        graphon = gaussian_filter(graphon, sigma=0.3)
        graphon = np.clip(graphon, 0, 1)
        
        return graphon, node_positions
    
    def compute_metrics(self, graphon: np.ndarray, W: np.ndarray) -> Dict:
        """Compute graphon metrics."""
        # Integrated connectivity
        integrated_connectivity = np.mean(graphon)
        
        # Entropy
        p = graphon.flatten()
        p = p[p > 1e-6]
        if len(p) > 0:
            entropy = -np.sum(p * np.log(p + 1e-8)) / (np.log(len(p) + 1) + 1e-8)
        else:
            entropy = 0
        
        # Core-periphery
        degrees = W.sum(axis=1)
        if len(degrees) > 0:
            core_threshold = np.percentile(degrees, 80)
            core_nodes = degrees > core_threshold
            core_ratio = np.sum(core_nodes) / len(core_nodes)
            
            if np.sum(core_nodes) > 0 and np.sum(~core_nodes) > 0:
                core_density = np.mean(W[core_nodes][:, core_nodes])
                periphery_density = np.mean(W[~core_nodes][:, ~core_nodes])
                core_periphery_ratio = core_density / (periphery_density + 1e-8)
            else:
                core_density = 0
                periphery_density = 0
                core_periphery_ratio = 1
        else:
            core_ratio = 0
            core_density = 0
            periphery_density = 0
            core_periphery_ratio = 1
        
        # Modularity approximation
        modularity = np.var(graphon) / (np.mean(graphon) + 1e-8)
        
        return {
            "integrated_connectivity": float(integrated_connectivity),
            "entropy": float(entropy),
            "core_ratio": float(core_ratio),
            "core_density": float(core_density),
            "periphery_density": float(periphery_density),
            "core_periphery_ratio": float(core_periphery_ratio),
            "modularity": float(modularity),
            "core_count": int(np.sum(core_nodes)) if len(degrees) > 0 else 0,
            "total_nodes": len(degrees)
        }
    
    def detect_phase_transition(self) -> Dict:
        """Detect phase transitions from history."""
        if len(self.metrics_history) < 3:
            return {
                "transition_detected": False,
                "phase_type": "stable",
                "confidence": 0
            }
        
        # Get recent metrics
        recent = self.metrics_history[-5:]
        
        # Check for significant changes
        connectivity = [m["integrated_connectivity"] for m in recent]
        core_ratio = [m["core_ratio"] for m in recent]
        
        if len(connectivity) >= 2:
            conn_change = connectivity[-1] - connectivity[-2]
            core_change = core_ratio[-1] - core_ratio[-2]
            
            # Detect phase type
            if abs(conn_change) > self.transition_threshold:
                if conn_change > 0:
                    phase_type = "connectivity_buildup"
                else:
                    phase_type = "connectivity_breakdown"
                transition_detected = True
                confidence = min(1.0, abs(conn_change) / self.transition_threshold)
            elif abs(core_change) > self.transition_threshold * 0.5:
                if core_change > 0:
                    phase_type = "core_periphery_formation"
                else:
                    phase_type = "core_periphery_dissolution"
                transition_detected = True
                confidence = min(1.0, abs(core_change) / self.transition_threshold)
            else:
                phase_type = "stable"
                transition_detected = False
                confidence = 1.0 - abs(conn_change) / (self.transition_threshold + 1e-8)
        else:
            phase_type = "stable"
            transition_detected = False
            confidence = 0
        
        return {
            "transition_detected": transition_detected,
            "phase_type": phase_type,
            "confidence": float(confidence),
            "conn_change": float(conn_change) if len(connectivity) >= 2 else 0,
            "core_change": float(core_change) if len(core_ratio) >= 2 else 0
        }
    
    def analyze_universe(self, returns: np.ndarray, macro: np.ndarray, 
                         tickers: List[str]) -> Dict:
        """Full analysis."""
        n_samples, n_etfs = returns.shape
        
        if n_samples < 50:
            return {"error": "Insufficient data"}
        
        # Build adjacency
        W = self.build_adjacency_matrix(returns, macro)
        
        # Estimate graphon
        graphon, node_positions = self.estimate_graphon(W)
        
        # Compute metrics
        metrics = self.compute_metrics(graphon, W)
        
        # Store history
        self.W_history.append(W)
        self.metrics_history.append(metrics)
        
        # Detect phase
        phase = self.detect_phase_transition()
        
        # Identify core ETFs
        degrees = W.sum(axis=1)
        if len(degrees) > 0:
            core_threshold = np.percentile(degrees, 80)
            core_indices = np.where(degrees > core_threshold)[0]
            core_etfs = [tickers[i] for i in core_indices]
        else:
            core_etfs = []
        
        # Pick ETFs
        # In stable phase: pick high degree (core)
        # In transition: pick low degree (non-core, less exposed)
        if phase.get("transition_detected", False) and phase.get("phase_type") != "stable":
            # During transition, pick stable ETFs
            degree_rank = np.argsort(degrees)[:self.config.get("TOP_N", 3)]
        else:
            # Stable: pick core ETFs
            degree_rank = np.argsort(degrees)[-self.config.get("TOP_N", 3):][::-1]
        
        picks = []
        for idx in degree_rank[:self.config.get("TOP_N", 3)]:
            ticker = tickers[idx]
            expected_return = returns[-5:, idx].mean() * 100
            is_core = idx in core_indices if len(core_indices) > 0 else False
            
            if is_core:
                confidence = "High" if expected_return > 0.2 else "Medium"
            else:
                confidence = "Medium" if expected_return > 0.2 else "Low"
            
            picks.append({
                "ticker": ticker,
                "expected_return": round(expected_return, 2),
                "confidence": confidence,
                "is_core": is_core
            })
        
        return {
            "picks": picks,
            "graphon": graphon.tolist(),
            "metrics": metrics,
            "phase": phase,
            "core_etfs": core_etfs,
            "node_positions": node_positions.tolist() if len(node_positions) > 0 else []
        }
