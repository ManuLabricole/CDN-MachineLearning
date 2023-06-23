import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

from statsmodels.graphics.tsaplots import plot_pacf


class ArimaPredictor:
    def __init__(self):
        self.data = None
        self.print_created()
     
    def is_initialized(self):
        return True
    
    def print_created(self):
        print("[ArimaPredictor] Created")
           
    def load_data(self, dataframe: pd.DataFrame):
        if type(dataframe) == pd.DataFrame:
            print("[ArimaPredictor] DataFrame loaded...")
            self.data = dataframe
        else:
            print("[ArimaPredictor] Data is not a DataFrame")
            self.data = None
            
    def check_data_stationarity(self, method:str="all" ,significance_level:float=0.05, data:pd.DataFrame=None):
        
        if data is not None:
            self.load_data(data)
        
        print("[ArimaPredictor] Checking data stationarity...")
        if self.data is None:
            print("[ArimaPredictor] Data is not loaded")
            raise Exception("Data is not loaded")
        else:
            if method == "all":
                assess_stationarity_with_adf(self.data, significance_level)
                assess_stationarity_with_kpss(self.data, significance_level)
            elif method == "adfuller":
                assess_stationarity_with_adf(self.data, significance_level)
            elif method == "kpss":
                assess_stationarity_with_kpss(self.data, significance_level)
            else:
                print("[ArimaPredictor] Method not recognized")
                raise Exception("Method not recognized")
        
    def plot_data_curves(self, data:pd.DataFrame=None, color_palette:str="tab20_r", rolling_1:int=6, rolling_2:int=12):
        
        if data is not None:
            self.load_data(data)
        
        print("[ArimaPredictor] Plotting data curves...")
        print("[ArimaPredictor] Rolling 1: ", rolling_1)
        print("[ArimaPredictor] Rolling 2: ", rolling_2)
        
        study_stationarity(self.data, rolling_1, rolling_2, color_palette)

    def plot_ACF_PACF(self, data:pd.DataFrame=None, lags:int=50, color_palette:str="tab20_r"):
        
        if data is not None:
            self.load_data(data)
        elif data is None and self.data is None:
            print("[ArimaPredictor] Data is not loaded")
            raise Exception("Data is not loaded")
        elif data is None and self.data is not None:
            pass
        
        print("[ArimaPredictor] Plotting ACF and PACF...")
        print("[ArimaPredictor] Lags: ", lags)
        
        
        plot_acf(self.data, lags=lags)
        plot_pacf_perso(self.data, lags=lags)
        

# End Class ArimaPredictor

# Utils functions

# STATIONARITY TESTS

def assess_stationarity_with_adf(data, significance_level=0.05):
    print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")  
    result = adfuller(data)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] < significance_level:
        print("Data is stationary")
    else:
        print("Data is not stationary")
    print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°\n")  

def assess_stationarity_with_kpss(data, significance_level=0.05):
    print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")  
    statistic, p_value, n_lags, critical_values = kpss(data, regression='c')
    # regression='c' indicates constant is used in the test equation.
    # For testing around a trend, use 'ct'
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critical values:')
    for key, value in critical_values.items():
        print(f'\t{key} : {value}')
    if p_value < significance_level:
        print("Data is not stationary")
    else:
        print("Data is stationary")
    print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°\n")  
    
def study_stationarity(dff:pd.DataFrame, rolling_1:int=6, rolling_2:int=12, color_palette:str="tab20_r"):
    
    colors = sns.color_palette(color_palette, 16)
    
    passengers = dff["Passengers"]
    passengersRolling = dff['Passengers'].rolling(window=rolling_1).mean()
    passengersRolling2 = dff['Passengers'].rolling(window=rolling_2).mean()
    
    x = np.arange(len(passengersRolling2.dropna()))
    y = passengersRolling2.dropna()
    coefficients = np.polyfit(x, y, 1)
    a = coefficients[0]  # Slope
    b = coefficients[1]  # Intercept
    regression_curve = a * x + b
    
    print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
    print('Coefficient b:', b)
    print('Coefficient a:', a)
    print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
    print("EXPECTING --> a = 0 <-- for stationarity")
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16,9))
    fig.suptitle("Passengers Evolution Tendency", fontsize=20)

    # Plotting the normal distribution curve
    sns.lineplot(
        ax=ax,
        x = dff.index,
        y = passengers,
        color = colors[1],
        label="Passengers Evolution"
    )
    sns.lineplot(
        ax=ax,
        x = dff.index,
        y = passengersRolling,
        color = colors[6],
        label=f"Rolling {rolling_1}",
        linewidth=2
    )
    sns.lineplot(
        ax=ax,
        x = dff.index,
        y = passengersRolling2,
        color = colors[4],
        label=f"Rolling {rolling_2}",
        linewidth=2
    )
    
    # Displaying the plot
    plt.show()
    
    
    passengersVar = dff["Passengers"].var()
    passengersRollingVar = dff['Passengers'].rolling(window=rolling_1).var()
    passengersRollingVar2 = dff['Passengers'].rolling(window=rolling_2).var()
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16,9))
    fig.suptitle("Variance Evolution Tendency", fontsize=20)
    
    
    # Plotting the normal distribution curve
    sns.lineplot(
        ax=ax,
        x = dff.index,
        y = passengersVar,
        color = colors[1],
        label="Variance Evolution"
    )
    sns.lineplot(
        ax=ax,
        x = dff.index,
        y = passengersRollingVar,
        color = colors[6],
        label=f"Rolling Variance {rolling_1}",
        linewidth=2
    )
    sns.lineplot(
        ax=ax,
        x = dff.index,
        y = passengersRollingVar2,
        color = colors[4],
        label=f"Rolling Variance {rolling_2}",
        linewidth=2
    )
    
    # Displaying the plot
    plt.show()
    
# ACF and PACF plots:

def plot_pacf_perso(data:pd.DataFrame, lags:int=24):
    
    print("[INFO] Plotting PACF...")
    print(type(data))
    
    try:
        df = data.copy()
        cols = df.columns
    except:
        raise Exception("[ERROR]")

    
    plot_pacf(df[cols[0]])
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.title('Partial Autocorrelation Function (PACF) - log / diff ')
    plt.show()
    

def plot_acf(data:pd.DataFrame, lags:int=24):
    print("[INFO] Plotting ACF...")
    
    df = data.copy()
    
    if type(df) == pd.core.frame.DataFrame or type(df) == pd.core.series.Series:
        cols = df.columns
        df = df[cols[0]]
        print("[WARNING] Dataframe detected, using first column")
    else:
        print("[INFO] Dataframe not detected, using data as is")
        print("[INFO] Data type:", type(df))
        print("[INFO] Trying to plot...")
    
    # Calculate the ACF
    acf = stattools.acf(df, nlags=24)
    # Plot the ACF
    plt.plot(acf)
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='gray')
    plt.show()