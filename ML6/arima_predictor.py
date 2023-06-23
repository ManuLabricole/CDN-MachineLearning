import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


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
            
    def check_data_stationarity(self, method:str="all" ,significance_level:float=0.05):
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
        
        
# End Class ArimaPredictor

# Utils functions

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