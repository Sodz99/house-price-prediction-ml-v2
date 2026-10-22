from sklearn.preprocessing import StandardScaler

# Function to perform standard scaling 

def std_scaler(df):
  scaler = StandardScaler()
  scaler.fit(df)
  df = scaler.transform(df)
  return df