import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


def nullify_zero(data):
    # replace zeros with NANs
    data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
    return data


def impute_median(data, var):
    temp = data[data[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median()
    data.loc[(data['Outcome'] == 0) & (data[var].isnull()), var] = temp.loc[0, var]
    data.loc[(data['Outcome'] == 1) & (data[var].isnull()), var] = temp.loc[1, var]
    return data


def impute_values(data):
    # impute values using the function
    data = impute_median(data, 'Glucose')
    data = impute_median(data, 'BloodPressure')
    data = impute_median(data, 'SkinThickness')
    data = impute_median(data, 'Insulin')
    data = impute_median(data, 'BMI')
    return data


def scale_values(x):
    # scale the values using a StandardScaler
    scaler = StandardScaler()
    scaler = scaler.fit(x)
    X = scaler.transform(x)
    dump(scaler, 'scaler.joblib')
    return X
