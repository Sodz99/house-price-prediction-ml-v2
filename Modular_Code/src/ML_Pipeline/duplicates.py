
# Function to remove duplicate values present in columns 

def remove_duplicates(df, var):
    if var in df : 
        df[df.duplicated(var)]
        df.drop_duplicates(subset=var, keep='first', inplace=True)
        return df
    else :
        raise ValueError ("Column not present")