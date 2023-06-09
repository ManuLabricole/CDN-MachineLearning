import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.stattools as stattools

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from pmdarima.arima import ARIMA

from statsmodels.graphics.tsaplots import plot_pacf

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


class ArimaPredictor:
    def __init__(self):
        self.data = None
        self.print_created()
     
    def is_initialized(self):
        return True
    
    def data(self):
        return self.data
    
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
        
        print("[ArimaPredictor] Plotting data curves...")
        print("[ArimaPredictor] Rolling 1: ", rolling_1)
        print("[ArimaPredictor] Rolling 2: ", rolling_2)
        
        study_stationarity(self.data, rolling_1, rolling_2, color_palette)

    def plot_ACF_PACF(self, data:pd.DataFrame=None, lags:int=24, color_palette:str="tab20_r"):
        """
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
        """
        
        if data is None:
            data = self.data
        
        n_sample = len(data)
        lag_acf = acf(np.log(data).diff().dropna(), nlags=lags)
        lag_pacf = pacf(np.log(data).diff().dropna(), nlags=lags)

        pct_95 = 1.96/np.sqrt(n_sample)

        plt.figure(figsize=(15, 3))
        #Plot ACF:
        plt.subplot(121)
        plt.stem(lag_acf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-pct_95, linestyle='--', color='gray')
        plt.axhline(y=pct_95, linestyle='--', color='gray')
        # plt.axvline(x=q, color='black', linestyle='--', label=f'q={q}')
        # plt.legend()
        plt.title('Autocorrelation Function')            

        #Plot PACF
        plt.subplot(122)
        plt.stem(lag_pacf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-pct_95, linestyle='--', color='gray') # represent 95 % of a gaussian data
        plt.axhline(y=pct_95, linestyle='--', color='gray') # represente 95 % of a gaussian data
        # plt.axvline(x=p, color='black', linestyle='--', label=f'p={p}')
        plt.title('Partial Autocorrelation Function')
        # plt.legend()
        plt.show()
        
    def ts_train_test_split(self, data=None, split_date="1959-01-01"):
        '''
        Split time series into training and test data

        Parameters:
        -------
        data - pd.DataFrame - time series data.  Index expected as datatimeindex
        split_date - the date on which to split the time series

        Returns:
        --------
        tuple (len=2) 
        0. pandas.DataFrame - training dataset
        1. pandas.DataFrame - test dataset
        '''
        
        if data is None:
            print("[ArimaPredictor] Data not sent")
            print("[ArimaPredictor] Loading data...")
            data = self.data
            cols = data.columns
            data = data[cols[0]]
        else:
            pass
        
        train = data.loc[data.index < split_date]
        test = data.loc[data.index >= split_date]
        
        self.train = train
        self.test = test
        
        return train, test
    
    def fit(self, data, plot:bool=True, seasonal:bool=True, transform_log:bool=True, p:int=1, d:int=1, q:int=1, P:int=1, D:int=1, Q:int=1, s:int=12):
        print("[ArimaPredictor] Fitting model...")
        print("[ArimaPredictor] p: ", p)
        print("[ArimaPredictor] d: ", d)
        print("[ArimaPredictor] q: ", q)
        print("[ArimaPredictor] P: ", P)
        print("[ArimaPredictor] D: ", D)
        print("[ArimaPredictor] Q: ", Q)
        print("[ArimaPredictor] s: ", s)
        
        if transform_log:
            print("[ArimaPredictor] Transforming data to log...")
            data = np.log(data)
            
        if seasonal:
            self.model = ARIMA(
            order=(p, d, q),
            seasonal_order=(P, D, Q, s), # 12 for monthly data
            suppress_warnings = True
        )
        else:
            self.model = ARIMA(
                order = (p, d, q),
                suppress_warnings = True
            )
            
        self.model.fit(data)
        
        if plot:
            self.model.plot_diagnostics(figsize=(12,9));
            plt.show()
        
        print("[ArimaPredictor] Model fitted")
        
    def predict(self, train, test, plot:bool=True):
        print("[ArimaPredictor] Predicting...")
        print("[ArimaPredictor] n_periods: ", len(test))
    
        y_pred, conf_int = self.model.predict(n_periods=len(test), return_conf_int=True)

        if plot:
            plt.plot(train, label='Train')
            plt.plot(test, label='Test')
            
            plt.fill_between(y_pred.index,
                         *np.exp(conf_int).T,
                         alpha=0.2, color='orange',
                         label="ARIMA Confidence Intervals")
        
            plt.plot(np.exp(y_pred), label='ARIMA Prediction')
        
            plt.legend()
        
        self.y_pred = y_pred
        print("[ArimaPredictor] Prediction done")  
        
    def get_metrics(self):
        if self.y_pred is None:
            print("[ArimaPredictor] No prediction done yet")
            return None
        else:
            mape = mean_absolute_percentage_error(self.test, np.exp(self.y_pred))
            rmse, rmse_mean = find_model_rmse(self.test, np.exp(self.y_pred))
            
            return mape, rmse, rmse_mean
        
    def grid_search(self, range_to_study = 1):
        R = range_to_study + 1
        test_number = 0

        param_tested = []
        MAPE_LIST = []
        RMSE_LIST = []
        RMSE_MEAN_LIST = []

        for p in range(R):
            for q in range(R):
                for d in range(R):
                    for P in range(R):
                        for Q in range(R):
                            for D in range(R):

                                print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
                                print(f"Testing --> {test_number} / {R*R*R*R*R*R}")
                                print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

                                param = {
                                    "p" : p,
                                    "q" : q,
                                    "d" : d,
                                    "P" : P,
                                    "Q" : Q,
                                    "D" : D,
                                }

                                self.fit(self.train, plot=False, **param)
                                self.predict(self.train, self.test, plot=False)
                                MAPE, RMSE, RMSE_MEAN = self.get_metrics()

                                param_tested.append(param)
                                MAPE_LIST.append(MAPE)
                                RMSE_LIST.append(RMSE)
                                RMSE_MEAN_LIST.append(RMSE_MEAN)

                                test_number += 1
                                
        self.grid_search_results = pd.DataFrame({
            "param_tested" : param_tested,
            "MAPE" : MAPE_LIST,
            "RMSE" : RMSE_LIST,
            "RMSE_MEAN" : RMSE_MEAN_LIST
        })
        
            

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
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
    
# COMPUTING THE  METRICS

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print("MAPE", mape)
    return mape

def find_model_rmse(test_data, y_pred_transformed):
    #Calculate RMSE and RMSE/mean(test)
    RMSE = np.sqrt(mean_squared_error(test_data, y_pred_transformed))
    RMSE_MEAN = RMSE/np.mean(test_data)
    print(f"RMSE --> {RMSE}")
    print(f"RMSE/MEAN --> {RMSE/np.mean(test_data)}")
    return RMSE, RMSE_MEAN