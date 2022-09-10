# Financial Signal Processing (Fall 2020)

Prof: Fred Fontaine

This course approaches financial engineering from a signal processing perspective. Stochastic processes: random walks, Brownian motion, Ito calculus, continuous models
including Black- Scholes, discrete models including negative binomial, martingales, stopping times. Representation and analysis of financial concepts such as price, risk,
volatility, futures, options, arbitrage, derivatives, portfolios and trading strategies. Analysis of single and multiple nonstationary time series, GARCH models.
Optimization methods, big data and machine learning in finance.



# Submissions - 4 Modeling Assignments: 
### 1.  Binomial Asset Pricing Model \& Put-Call Combinations
##### [FSP_PS1.ipynb](https://github.com/yuvalofek/Financial-Signal-Processing/blob/main/FSP_PS1.ipynb)
* Modeled Combinations: Calls, Puts, Digital Calls, Digital Puts, Straddle, Call-Put Spread, Butterfly, Call Ladder, Digital Call Spread 
* Binomial Asset Pricing: Exhaustive path generation using arrays, Exhaustive path generation using recursion, Monte Carlo simulation, Path dependent analysis 


### 2. Markowitz Portfolio Analysis
##### [FSP_PS2.ipynb](https://github.com/yuvalofek/Financial-Signal-Processing/blob/main/FSP_PS2.ipynb)
* Performed yearly Markowitz Portfolio Analysis - Efficient frontier, MVP point (with and without shorting), Market point (with and without shorting)
* Analyzed correlation, beta, systematic risk, and diversifiable risk
* Conducted eigenvalue analysis on the covariance matrix. 
##### Data: 
* Farma-French, http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
* S\&P500, https://finance.yahoo.com/
* USD LIBOR rate for 1 year maturity, https://www.macrotrends.net/2515/1-year-libor-rate-historical-chart
 
### 3. Stocastic Differential Equations for Finance
##### [FSP3_Stochastic Calc.ipynb](https://github.com/yuvalofek/Financial-Signal-Processing/blob/main/FSP3_Stochastic_Calc.ipynb) 
* Simulated Brownian Motion
* Implemented Black Scholes and compared expected call value mid with Monte Carlo Simulations
* Generated plot of half-path call value based Black Scholes model
* Created stochastic paths based on the Cox-Ingersoll-Ross interest rate model and compared expected mean and variance to theoretical values.

### 4. Time Series Modeing using GARCH, ARCH, ARMA, and AR
```
PS4 Folder
|-- AAPL (1).csv 
|-- AAPL (2).csv
|-- ADBE (1).csv
|-- ADBE (2).csv
|-- ^SP500TR (1).csv
|-- ^SP500TR (2).csv
|-- FSP_4_Data.mat
'-- FSP4_TImeModeling.m 
```

[FSP4_TImeModeling.m](https://github.com/yuvalofek/Financial-Signal-Processing/blob/main/PS4/FSP4_TImeModeling.m)
* Generating Gaussian, Cauchy, \& Student t (t=5 \& 5=10) distributions
* Synthesized ARMA data and computed AR coefficients using least-squares fit 
  * Compared Cholesky factorization results to Levinson-Durbin 
* Simulated models with unit root nonstationarity and used partial correlation coefficients $\gamma(m) / \gamma(0)$ to detect it
* Implemented an example of cointegration of two signals
* Conducted and ARCH/GARCH analysis on the synthesized and the provided data (see bellow) 


Data Files: (from https://finance.yahoo.com/)
* AAPL (1).csv
* AAPL (2).csv
* ADBE (1).csv
* ADBE (2).csv
* ^SP500TR (1).csv
* ^SP500TR (2).csv 

Alternative Data File (replaces uploading on part 6)
* FSP_4_Data.mat
