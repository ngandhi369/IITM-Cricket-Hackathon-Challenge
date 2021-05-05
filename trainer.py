import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.read_csv('myPreprocessed.csv')

# Encoding means - categorical values mapped to integer values
venue_encode = LabelEncoder()
team_encode = LabelEncoder()

data['venue'] = venue_encode.fit_transform(data['venue'])
data['batting_team'] = team_encode.fit_transform(data['batting_team'])
data['bowling_team'] = team_encode.fit_transform(data['bowling_team'])

anArray = data.to_numpy()

# print(anArray[0][0])
# print(anArray[0][1])
# print(anArray[0][2])
# print(anArray[0][3])
# print(anArray[0][4])

X,y = anArray[:,:4], anArray[:,4]
# X -> all row, 0 to 3 col
# y -> all row, 4th col

X = np.concatenate((np.eye(42)[anArray[:,0]],
                    np.eye(2)[anArray[:,1] -1 ],
                    np.eye(15)[anArray[:,2]],
                    np.eye(15)[anArray[:,3]],
                    ), axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

linearRegressor = LinearRegression()

linearRegressor.fit(X_train, y_train)

joblib.dump(linearRegressor, 'regression_model.joblib')
joblib.dump(venue_encode, 'venue_encoder.joblib')
joblib.dump(team_encode, 'team_encoder.joblib')

print(linearRegressor.score(X_test, y_test))