import numpy as np
import pandas as pd

def generate_drifted_data():
    # จำลองราคาบ้านที่สูงขึ้นผิดปกติ (Data Drift)
    normal_data = np.random.normal(loc=50000, scale=10000, size=100)
    drifted_data = np.random.normal(loc=90000, scale=15000, size=100)
    
    df_drift = pd.DataFrame({'house_price': drifted_data})
    df_drift.to_csv('data/current_batch.csv', index=False)
    print("Simulated drifted data saved!")

if __name__ == '__main__':
    generate_drifted_data()