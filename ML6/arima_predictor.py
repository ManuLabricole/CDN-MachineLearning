import pandas as pd


class ArimaPredictor:
    def __init__(self):
        self.data = None
        self.print_created()
        
    def load_data(self, dataframe: pd.DataFrame):
        if type(dataframe) == pd.DataFrame:
            print("[ArimaPredictor] DataFrame loaded...")
            self.data = dataframe
        else:
            print("[ArimaPredictor] Data is not a DataFrame")
            self.data = None
        
    def print_created(self):
        return f"[ArimaPredictor] Created"