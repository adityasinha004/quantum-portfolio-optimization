import yfinance as yf
import numpy as np
import pandas as pd

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import LinearEqualityToPenalty
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

def main():
    print("=== S1: Download Dataset & Observe Features ===")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    print(f"Downloading data for {tickers}...")
    
    # Download dataset
    data = pd.DataFrame()
    for t in tickers:
        df = yf.download(t, start="2023-01-01", end="2024-01-01", progress=False)
        # Newer yfinance versions return a MultiIndex dataframe when downloading even a single ticker.
        # We extract 'Close' or 'Adj Close' safely.
        if isinstance(df.columns, pd.MultiIndex):
            data[t] = df[('Close', t)]
        else:
            data[t] = df['Close']
        
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Split data to calculate "Train" and "Test" metrics (approximating Train/Test Acc concepts)
    split_idx = int(len(returns) * 0.7)
    train_returns = returns.iloc[:split_idx]
    test_returns = returns.iloc[split_idx:]
    
    # Features: Expected return (f1) and Risk/Covariance (f2)
    mu_train = train_returns.mean().values * 252  # Annualized expected return
    sigma_train = train_returns.cov().values * 252  # Annualized risk (covariance)
    
    print("\nExpected Returns (f1 - Train):")
    for i, t in enumerate(tickers):
        print(f"  {t}: {mu_train[i]:.4f}")
        
    print("\nCovariance Matrix (Risk/f2 - Train):")
    print(np.round(sigma_train, 4))

    print("\n=== S2 & S3: QUBO Formulation (Multi-Objective) ===")
    qp = QuadraticProgram()
    
    for ticker in tickers:
        qp.binary_var(ticker)
        
    # Objective: Minimize Risk - q * Return
    # q is the risk tolerance factor (0 to 1)
    q = 0.5 
    
    linear_dict = {tickers[i]: -q * mu_train[i] for i in range(len(tickers))}
    quadratic_dict = {}
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            quadratic_dict[(tickers[i], tickers[j])] = (1-q) * sigma_train[i, j]
            
    qp.minimize(linear=linear_dict, quadratic=quadratic_dict)
    
    # Budget constraint: select exactly K stocks (e.g., K=2)
    K = 2
    linear_constraint = {ticker: 1 for ticker in tickers}
    qp.linear_constraint(linear=linear_constraint, sense='==', rhs=K, name='budget')
    
    # Convert constraint to penalty to form Unconstrained QUBO
    lineq2penalty = LinearEqualityToPenalty()
    qubo = lineq2penalty.convert(qp)
    print("\nQUBO Objective (after penalty):")
    print(qubo.objective)

    print("\n=== S4: Train QAOA Algorithm ===")
    print("Initializing QAOA on local simulator...")
    # NOTE: To use real IBMQ, register at https://quantum.ibm.com/ and replace Sampler() below
    # with an IBMQ backend sampler using qiskit_ibm_runtime.
    
    sampler = Sampler() # Local IBM-like Simulator
    cobyla = COBYLA(maxiter=100)
    qaoa = QAOA(sampler=sampler, optimizer=cobyla, reps=2)
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    
    print("Solving...")
    result = qaoa_optimizer.solve(qubo)
    
    print("\nOptimal Portfolio Configuration (Binary Result):", result.x)
    selected_stocks = [tickers[i] for i in range(len(tickers)) if result.x[i] == 1.0]
    print("Selected stocks:", selected_stocks)

    print("\n=== S5: Results ===")
    # Calculate portfolio values assuming equal weight among selected
    weights = result.x / sum(result.x) if sum(result.x) > 0 else result.x
    
    train_port_return = np.dot(weights, mu_train)
    train_port_risk = np.dot(weights.T, np.dot(sigma_train, weights))
    
    mu_test = test_returns.mean().values * 252
    sigma_test = test_returns.cov().values * 252
    
    test_port_return = np.dot(weights, mu_test)
    test_port_risk = np.dot(weights.T, np.dot(sigma_test, weights))
    
    print(f"Train Portfolio Expected Return: {train_port_return:.4f}")
    print(f"Train Portfolio Risk (Variance): {train_port_risk:.4f}")
    print(f"Test Portfolio Expected Return:  {test_port_return:.4f}")
    print(f"Test Portfolio Risk (Variance):  {test_port_risk:.4f}")
    
    if train_port_risk > 0:
        print(f"Train Sharpe Ratio (Acc proxy): {train_port_return / np.sqrt(train_port_risk):.4f}")
    if test_port_risk > 0:
        print(f"Test Sharpe Ratio (Acc proxy):  {test_port_return / np.sqrt(test_port_risk):.4f}")

if __name__ == "__main__":
    main()