# Financial Signal Processing (Fall 2020)

Prof: Fred Fontaine

This course approaches financial engineering from a signal processing perspective. Stochastic processes: random walks, Brownian motion, Ito calculus, continuous models
including Black- Scholes, discrete models including negative binomial, martingales, stopping times. Representation and analysis of financial concepts such as price, risk,
volatility, futures, options, arbitrage, derivatives, portfolios and trading strategies. Analysis of single and multiple nonstationary time series, GARCH models.
Optimization methods, big data and machine learning in finance.



# Submissions:
4 Modeling Assignments:
### Binomial Asset Pricing Model \& Put-Call Combinations
FSP_PS1.ipynb 
* Modeled Combinations: Calls, Puts, Digital Calls, Digital Puts, Straddle, Call-Put Spread, Butterfly, Call Ladder, Digital Call Spread 
* Binomial Asset Pricing: Exhaustive path generation using arrays, Exhaustive path generation using recursion, Monte Carlo simulation, Path dependent analysis 


### Markowitz Portfolio Analysis
FSP_PS2.ipynb 
* Performed yearly Markowitz Portfolio Analysis - Efficient frontier, MVP point (with and without shorting), Market point (with and without shorting)
* Analyzed correlation, beta, systematic risk, and diversifiable risk
* Conducted eigenvalue analysis on the covariance matrix. 
Data: 
* Farma-French, http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
* S\&P500, https://finance.yahoo.com/
* USD LIBOR rate for 1 year maturity, https://www.macrotrends.net/2515/1-year-libor-rate-historical-chart
 
### Stocastic Differential Equations for Finance
FSP3_Stochastic Calc.ipynb 
* Simulated Brownian Motion
* Implemented Black Scholes and compared expected call value mid with Monte Carlo Simulations
* Generated plot of half-path call value based Black Scholes model
* Created stochastic paths based on the Cox-Ingersoll-Ross interest rate model and compared expected mean and variance to theoretical values.

### Time Series Modeing using GARCH, ARCH, ARMA, and AR
PS4 Folder Content
  * Data files: AAPL (1).csv, AAPL (2).csv, ADBE (1).csv, ADBE (2).csv, ^SP500TR (1).csv, ^SP500TR (2).csv
  * FSP4_TImeModeling.m 



