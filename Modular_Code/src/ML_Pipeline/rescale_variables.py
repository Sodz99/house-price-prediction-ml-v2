
# Function to variables that need to be rescaled / incorrectly scaled variables


def rescale_variables(df, column_list, value):
    if value !=0:
        for var in column_list:
            df[var] = (df[var]) / value
    
        return df 
    else :
        raise ValueError("Value not acceptable")

