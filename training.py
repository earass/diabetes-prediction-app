from utils import read_dataset
from preprocessing import nullify_zero, impute_values, scale_values
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

data = read_dataset()

data = nullify_zero(data)

data = impute_values(data)

# separate features and target as x & y
y = data['Outcome']
x = data.drop('Outcome', axis=1)
columns = x.columns

X = scale_values(x)

# features DataFrame
features = pd.DataFrame(X, columns=columns)

# split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# define the model
model = RandomForestClassifier(n_estimators=300, bootstrap=True, max_features='sqrt')

# fit model to training data
model.fit(x_train, y_train)

# predict on test data
y_pred = model.predict(x_test)

# evaluate performance
print(classification_report(y_test, y_pred))

dump(model, 'model.joblib')


# inference input data
pregnancies = 2
glucose = 13
bloodpressure = 30
skinthickness = 4
insulin = 5
bmi = 5
dpf = 0.55
age = 34
feat_cols = features.columns
row = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]

# transform the inference data same as training data
df = pd.DataFrame([row], columns=feat_cols)
X = scale_values(df)
features = pd.DataFrame(X, columns=feat_cols)

# make predictions using the already built model [0: healthy, 1:diabetes]
if model.predict(features) == 0:
    print("This is a healthy person!")
else:
    print("This person has high chances of having diabetes!")
