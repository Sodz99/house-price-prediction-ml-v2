from sklearn.preprocessing import LabelEncoder

# Function for performing label encoding for the categorical variables

def encode_categorical_variables(df, cat_vars):

    for i in range(len(cat_vars)):
        var = cat_vars[i]
        var_le = LabelEncoder()
        var_labels = var_le.fit_transform(df[var])
        var_mappings = {index: label for index, label in enumerate(var_le.classes_)}
        df[(var + '_labels')] = var_labels
        df.drop(columns=var, axis=1, inplace=True)
    return df
