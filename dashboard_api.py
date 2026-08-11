from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from typing import List
import yfinance as yf
import numpy as np
import pandas as pd
import io

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import LinearEqualityToPenalty
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

import os

# Safely import QiskitRuntimeService if installed
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    HAS_IBM_RUNTIME = True
except ImportError:
    HAS_IBM_RUNTIME = False
    QiskitRuntimeService = None
    SamplerV2 = None

app = FastAPI(
    title="Quantum Portfolio Optimization API", 
    description="S6 Dashboard API for QAOA-based portfolio optimization with graph visualizations"
)

IBM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN")
service = None
if HAS_IBM_RUNTIME:
    try:
        if IBM_TOKEN:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN)
            print("Successfully connected to IBM Quantum Runtime Service.")
        else:
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
            print("Successfully loaded saved IBM Quantum account.")
    except Exception as e:
        print(f"IBM Quantum Service not initialized. Falling back to local simulator. Reason: {e}")

class PortfolioRequest(BaseModel):
    tickers: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    risk_factor: float = 0.5
    budget: int = 2
    use_ibm_simulator: bool = False

class PortfolioResponse(BaseModel):
    selected_stocks: List[str]
    individual_expected_returns: dict
    individual_risks_variance: dict
    train_return: float
    train_risk: float
    train_sharpe: float
    test_return: float
    test_risk: float
    test_sharpe: float

@app.get("/", response_class=HTMLResponse)
def dashboard_home():
    """HTML Dashboard Overview for Top 5 Stocks Quantum Optimization."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quantum Portfolio Dashboard</title>
        <meta charset="utf-8">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 10px; }
            .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 25px; }
            .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }
            .card h3 { margin: 0 0 10px 0; color: #8b949e; font-size: 14px; text-transform: uppercase; }
            .card p { margin: 0; font-size: 22px; font-weight: bold; color: #3fb950; }
            .badge { background: #1f6feb; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; }
            .btn { display: inline-block; background: #238636; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: bold; }
            .btn:hover { background: #2ea043; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚛️ Quantum Portfolio Optimization API & Dashboard</h1>
            <p>Target Tickers: <span class="badge">AAPL</span> <span class="badge">MSFT</span> <span class="badge">GOOGL</span> <span class="badge">AMZN</span> <span class="badge">META</span></p>
            
            <div class="card-grid">
                <div class="card"><h3>Selected Stocks</h3><p id="sel">AMZN, META</p></div>
                <div class="card"><h3>Train Sharpe Ratio</h3><p id="tr-sharpe">3.19</p></div>
                <div class="card"><h3>Train Return</h3><p id="tr-ret">110.3%</p></div>
                <div class="card"><h3>Test Sharpe Ratio</h3><p id="te-sharpe">1.40</p></div>
                <div class="card"><h3>Test Return</h3><p id="te-ret">37.1%</p></div>
            </div>

            <div style="margin-top: 20px;">
                <a href="/docs" class="btn">Open Interactive API Docs (Swagger)</a>
                <a href="/plot" class="btn" style="background:#8957e5; margin-left: 10px;">View Top 5 Stocks Graph Visualization</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/optimize", response_model=PortfolioResponse)
def optimize_portfolio(request: PortfolioRequest):
    if request.budget > len(request.tickers):
        raise HTTPException(status_code=400, detail="Budget K cannot be greater than the number of tickers.")
        
    tickers = request.tickers
    
    # Download dataset
    data = pd.DataFrame()
    for t in tickers:
        df = yf.download(t, start=request.start_date, end=request.end_date, progress=False)
        if df.empty:
            raise HTTPException(status_code=400, detail=f"Failed to download data for ticker {t}.")
            
        if isinstance(df.columns, pd.MultiIndex):
            data[t] = df[('Close', t)]
        else:
            data[t] = df['Close']
            
    if data.empty or data.isna().all().all():
        raise HTTPException(status_code=400, detail="Failed to download data for the provided tickers and dates.")
        
    returns = data.pct_change().dropna()
    split_idx = int(len(returns) * 0.7)
    train_returns = returns.iloc[:split_idx]
    test_returns = returns.iloc[split_idx:]
    
    mu_train = train_returns.mean().values * 252
    sigma_train = train_returns.cov().values * 252
    
    # QUBO Formulation
    qp = QuadraticProgram()
    for ticker in tickers:
        qp.binary_var(ticker)
        
    q = request.risk_factor
    linear_dict = {tickers[i]: -q * mu_train[i] for i in range(len(tickers))}
    quadratic_dict = {}
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            quadratic_dict[(tickers[i], tickers[j])] = (1-q) * sigma_train[i, j]
            
    qp.minimize(linear=linear_dict, quadratic=quadratic_dict)
    
    linear_constraint = {ticker: 1 for ticker in tickers}
    qp.linear_constraint(linear=linear_constraint, sense='==', rhs=request.budget, name='budget')
    
    lineq2penalty = LinearEqualityToPenalty()
    qubo = lineq2penalty.convert(qp)
    
    if request.use_ibm_simulator and service and HAS_IBM_RUNTIME:
        try:
            backend = service.get_backend("ibmq_qasm_simulator")
            sampler = SamplerV2(backend=backend)
            print(f"Using IBM Cloud Simulator: {backend.name}")
        except Exception as e:
            print(f"Failed to use IBM Simulator, falling back to local. Error: {e}")
            sampler = Sampler()
    else:
        sampler = Sampler()
        
    cobyla = COBYLA(maxiter=100)
    qaoa = QAOA(sampler=sampler, optimizer=cobyla, reps=2)
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    
    result = qaoa_optimizer.solve(qubo)
    
    selected_stocks = [tickers[i] for i in range(len(tickers)) if result.x[i] == 1.0]
    weights = result.x / sum(result.x) if sum(result.x) > 0 else result.x
    
    train_port_return = float(np.dot(weights, mu_train))
    train_port_risk = float(np.dot(weights.T, np.dot(sigma_train, weights)))
    
    mu_test = test_returns.mean().values * 252
    sigma_test = test_returns.cov().values * 252
    
    test_port_return = float(np.dot(weights, mu_test))
    test_port_risk = float(np.dot(weights.T, np.dot(sigma_test, weights)))
    
    train_sharpe = train_port_return / np.sqrt(train_port_risk) if train_port_risk > 0 else 0.0
    test_sharpe = test_port_return / np.sqrt(test_port_risk) if test_port_risk > 0 else 0.0
    
    ind_returns = {tickers[i]: float(mu_train[i]) for i in range(len(tickers))}
    ind_risks = {tickers[i]: float(sigma_train[i][i]) for i in range(len(tickers))}
    
    return PortfolioResponse(
        selected_stocks=selected_stocks,
        individual_expected_returns=ind_returns,
        individual_risks_variance=ind_risks,
        train_return=train_port_return,
        train_risk=train_port_risk,
        train_sharpe=train_sharpe,
        test_return=test_port_return,
        test_risk=test_port_risk,
        test_sharpe=test_sharpe
    )

@app.get("/plot")
def get_graph_visualization():
    """Generates and serves PNG plot visualization for the top 5 stocks."""
    try:
        from dashboard import generate_matplotlib_dashboard, DEFAULT_DATA
        fig = generate_matplotlib_dashboard(DEFAULT_DATA, save_path=None)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate plot: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Quantum Portfolio Dashboard API on http://127.0.0.1:8000")
    uvicorn.run("dashboard_api:app", host="127.0.0.1", port=8000, reload=True)