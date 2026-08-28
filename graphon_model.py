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
    W(x,y) represents the continuous interaction probability between ETFs.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_bins = config.get("n_bins", 20)
        self.bandwidth = config.get("bandwidth", 0.1)
        self.transition_threshold = config.get("transition_threshold", 0.15)
        self.W_history = []
        self.phase_history = []
        self.entropy_history = []
        self.integrated_connectivity = []
        
    def build_adjacency_matrix(self, returns: np.ndarray, macro: np.ndarray) -> np.ndarray:
        """
        Build adjacency matrix W_t(i,j) from multiple sources.
        """
        n_samples, n_etfs = returns.shape
        
        # 1. Correlation matrix
        corr = np.corrcoef(returns.T)
        
        # 2. Volatility transmission (covariance)
        vol_cov = np.cov(returns.T)
        vol_cov_norm = vol_cov / (np.max(vol_cov) + 1e-8)
        
        # 3. Lead-lag relationships (cross-correlation at lag 1)
        lead_lag = np.zeros((n_etfs, n_etfs))
        for i in range(n_etfs):
            for j in range(n_etfs):
                if i != j:
                    cc = np.correlate(returns[:, i], returns[:, j], mode='full')
                    if len(cc) > 0:
                        max_lag = len(cc) // 2
                        if max_lag > 0:
                            # Look for maximum cross-correlation at lag 1
                            lag1 = cc[max_lag + 1] if max_lag + 1 < len(cc) else cc[max_lag]
                            lead_lag[i, j] = abs(lag1) / (np.std(returns[:, i]) * np.std(returns[:, j]) + 1e-8) * 0.5
        
        # 4. Sector exposure (if we had sector labels, but we'll use macro correlations)
        macro_corr = np.corrcoef(returns.T, macro.T)[:n_etfs, n_etfs:]
        sector_exposure = np.abs(macro_corr)
        
        # 5. Combined adjacency matrix
        W = (corr + vol_cov_norm + lead_lag) / 3
        W = np.clip(W, 0, 1)
        np.fill_diagonal(W, 0)
        
        # Enhance with sector exposure
        W = W * (1 + 0.2 * np.mean(sector_exposure, axis=1, keepdims=True))
        W = np.clip(W, 0, 1)
        
        return W
    
    def estimate_graphon(self, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate graphon W(x,y) from adjacency matrix.
        Uses stochastic block model approximation with binning.
        """
        n = W.shape[0]
        
        # Sort by degree to get node positions
        degrees = W.sum(axis=1)
        sorted_indices = np.argsort(degrees)[::-1]
        W_sorted = W[sorted_indices][:, sorted_indices]
        
        # Bin nodes into groups
        bin_size = max(1, n // self.n_bins)
        n_bins_actual = min(self.n_bins, n // bin_size)
        
        graphon = np.zeros((n_bins_actual, n_bins_actual))
        node_positions = np.zeros(n)
        
        for i in range(n):
            node_positions[i] = np.searchsorted(np.arange(0, n, bin_size), i) / n_bins_actual
        
        for i in range(n_bins_actual):
            for j in range(n_bins_actual):
                i_start = i * bin_size
                i_end = min((i + 1) * bin_size, n)
                j_start = j * bin_size
                j_end = min((j + 1) * bin_size, n)
                
                if i_start < i_end and j_start < j_end:
                    block = W_sorted[i_start:i_end, j_start:j_end]
                    graphon[i, j] = np.mean(block) if block.size > 0 else 0
        
        # Smooth graphon
        graphon = gaussian_filter(graphon, sigma=0.5)
        graphon = np.clip(graphon, 0, 1)
        
        return graphon, node_positions
    
    def compute_graphon_metrics(self, graphon: np.ndarray, W: np.ndarray) -> Dict:
        """
        Compute metrics from graphon.
        """
        # 1. Integrated connectivity
        integrated_connectivity = np.mean(graphon)
        
        # 2. Entropy (measure of uncertainty in connections)
        p = graphon.flatten()
        p = p[p > 0]
        if len(p) > 0:
            entropy = -np.sum(p * np.log(p + 1e-8)) / len(p)
        else:
            entropy = 0
        
        # 3. Core-periphery structure
        # Core: high-degree nodes
        degrees = W.sum(axis=1)
        core_threshold = np.percentile(degrees, 80)
        core_nodes = degrees > core_threshold
        core_density = np.mean(W[core_nodes][:, core_nodes]) if np.sum(core_nodes) > 0 else 0
        periphery_density = np.mean(W[~core_nodes][:, ~core_nodes]) if np.sum(~core_nodes) > 0 else 0
        
        # 4. Modularity (approximated from graphon)
        modularity = np.var(graphon) / (np.mean(graphon) + 1e-8)
        
        # 5. Spectral properties
        eigenvals = np.linalg.eigvalsh(graphon)
        spectral_gap = eigenvals[-1] - eigenvals[-2] if len(eigenvals) > 1 else 0
        
        return {
            "integrated_connectivity": float(integrated_connectivity),
            "entropy": float(entropy),
            "core_density": float(core_density),
            "periphery_density": float(periphery_density),
            "core_periphery_ratio": float(core_density / (periphery_density + 1e-8)),
            "modularity": float(modularity),
            "spectral_gap": float(spectral_gap),
            "core_ratio": float(np.sum(core_nodes) / len(core_nodes)),
        }
    
    def detect_phase_transition(self, metrics_history: List[Dict]) -> Dict:
        """
        Detect graphon phase transitions from history.
        """
        if len(metrics_history) < 2:
            return {"transition_detected": False, "type": "insufficient_data"}
        
        # Track changes in metrics
        integrated = [m["integrated_connectivity"] for m in metrics_history]
        entropy = [m["entropy"] for m in metrics_history]
        core_ratio = [m["core_ratio"] for m in metrics_history]
        
        # Detect sudden changes
        integrated_change = np.diff(integrated) if len(integrated) > 1 else [0]
        entropy_change = np.diff(entropy) if len(entropy) > 1 else [0]
        core_change = np.diff(core_ratio) if len(core_ratio) > 1 else [0]
        
        # Combined change signal
        change_signal = np.abs(integrated_change) + np.abs(entropy_change) + np.abs(core_change)
        
        if len(change_signal) > 0 and np.max(change_signal) > self.transition_threshold:
            transition_type = "unknown"
            if integrated_change[-1] < -self.transition_threshold / 3:
                transition_type = "connectivity_breakdown"
            elif integrated_change[-1] > self.transition_threshold / 3:
                transition_type = "connectivity_buildup"
            elif core_change[-1] > self.transition_threshold / 3:
                transition_type = "core_periphery_formation"
            elif core_change[-1] < -self.transition_threshold / 3:
                transition_type = "core_periphery_dissolution"
            
            return {
                "transition_detected": True,
                "type": transition_type,
                "magnitude": float(np.max(change_signal)),
                "direction": "increasing" if integrated_change[-1] > 0 else "decreasing"
            }
        
        return {"transition_detected": False, "type": "stable"}
    
    def analyze_universe(self, returns: np.ndarray, macro: np.ndarray, 
                         tickers: List[str]) -> Dict:
        """
        Full analysis of a universe.
        """
        n_samples, n_etfs = returns.shape
        
        if n_samples < 100:
            return {"error": "Insufficient data"}
        
        # Build adjacency matrix
        W = self.build_adjacency_matrix(returns, macro)
        
        # Estimate graphon
        graphon, node_positions = self.estimate_graphon(W)
        
        # Compute metrics
        metrics = self.compute_graphon_metrics(graphon, W)
        
        # Detect phase transitions
        self.W_history.append(W)
        self.phase_history.append(graphon)
        self.entropy_history.append(metrics["entropy"])
        self.integrated_connectivity.append(metrics["integrated_connectivity"])
        
        transition = self.detect_phase_transition(
            [{"integrated_connectivity": self.integrated_connectivity[i],
              "entropy": self.entropy_history[i],
              "core_ratio": metrics.get("core_ratio", 0)}
             for i in range(len(self.integrated_connectivity))]
        )
        
        # Identify ETFs in core
        degrees = W.sum(axis=1)
        core_threshold = np.percentile(degrees, 80)
        core_indices = np.where(degrees > core_threshold)[0]
        core_etfs = [tickers[i] for i in core_indices]
        
        # Pick ETFs based on transition state
        if transition.get("transition_detected", False):
            # In transition, pick stable ETFs (low degree change)
            degree_change = np.abs(np.diff(degrees) if len(degrees) > 0 else np.zeros_like(degrees))
            stable_indices = np.argsort(degree_change)[:self.config.get("TOP_N", 3)]
        else:
            # Stable state: pick highest degree (core) ETFs
            stable_indices = np.argsort(degrees)[-self.config.get("TOP_N", 3):][::-1]
        
        # Build picks
        picks = []
        for idx in stable_indices[:self.config.get("TOP_N", 3)]:
            ticker = tickers[idx]
            expected_return = returns[-5:, idx].mean() * 100
            
            # Confidence based on core status
            is_core = idx in core_indices
            confidence = "High" if is_core and expected_return > 0 else "Medium" if is_core else "Low"
            
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
            "transition": transition,
            "core_etfs": core_etfs,
            "node_positions": node_positions.tolist() if len(node_positions) > 0 else [],
            "phase_detected": transition.get("transition_detected", False),
            "phase_type": transition.get("type", "stable")
        }
