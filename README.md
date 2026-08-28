# P2-ETF-GRAPHON-SYSTEMIC

## Graphon-based ETF Network Analysis for Systemic Risk

### Concept

Instead of modelling discrete graph G=(V,E), this engine models a large network as a continuous function W(x,y) representing the probability/intensity of interaction between ETFs.

### Construction

W_t(i,j) is constructed from:
- Correlation
- Lead-lag relationships
- Volatility transmission
- Flows
- Sector exposure
- Macro sensitivity

### Outputs

- **Phase transitions**: Detects when the market network shifts from dispersed connectivity to concentrated core-periphery
- **Core ETFs**: Identifies the most connected ETFs (systemic importance)
- **Top picks**: ETFs with best expected return given network state

### Installation

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-GRAPHON-SYSTEMIC
cd P2-ETF-GRAPHON-SYSTEMIC
pip install -r requirements.txt# P2-ETF-GRAPHON-SYSTEMIC
