import polars as pl
import pandas as pd
import re

class Clean_Data:
    
    def __init__(self):
        pass
    
    def clean_csv(self, path : str , cols : list):
        data = pl.read_csv(path, columns=cols)
        activity_data = data.filter(pl.col('event_type').is_in(["cart", "purchase"]))
        new_path = re.sub(r'\.csv$', r'_clean.parquet', path)
        activity_data.write_parquet(new_path)
        
    def map_time_of_day(self, hour):
        if 5 <= hour <= 11:
            return "morning"
        elif 12 <= hour <= 16:
            return "afternoon"
        elif 17 <= hour <= 20:
            return "evening"
        else:
            return "night"
        
    def segment_user(self, num_purchases):
        if num_purchases == 0:
            return "window_shopper"
        elif num_purchases == 1:
            return "new_buyer"
        elif 2 <= num_purchases <= 5:
            return "repeat_buyer"
        else:
            return "loyal_buyer"
        
    def prepare_data(self, data):
        features_to_drop = ["event_time", "prev_event_time", "product_id", "user_id", "event_type", "hour_12"]
        data = data.drop(features_to_drop, axis=1)
        data_X = pd.get_dummies(data.drop(columns=["is_purchase"]), drop_first=True)
        data_y = data["is_purchase"]
        missing_count = data_X.isna().sum() > 0
        missing_count = missing_count[missing_count > 0]
        missing_count
        for column in missing_count.index:
            data_X[column] = data_X[column].fillna(data_X[column].median())
            
        return data_X, data_y
    