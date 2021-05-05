### Custom definitions and classes if any ###

import pandas as pd
import numpy as np
import joblib

def predictRuns(testInput):
    prediction = 0

    with open('regression_model.joblib', 'rb') as f:
        regressor = joblib.load(f)
    with open('venue_encoder.joblib', 'rb') as f:
        venue_encoder = joblib.load(f)
    with open('team_encoder.joblib', 'rb') as f:
        team_encoder = joblib.load(f)

    test_case = pd.read_csv(testInput)

    # t = test_case.to_numpy()
    # print(t[0])

    test_case['venue'] = venue_encoder.transform(test_case['venue'])
    test_case['batting_team'] = team_encoder.transform(test_case['batting_team'])
    test_case['bowling_team'] = team_encoder.transform(test_case['bowling_team'])

    test_case = test_case[['venue', 'innings', 'batting_team', 'bowling_team']]

    testArray = test_case.to_numpy()

    test_case = np.concatenate((np.eye(42)[testArray[:,0]],
                    np.eye(2)[testArray[:,1] -1 ],
                    np.eye(15)[testArray[:,2]],
                    np.eye(15)[testArray[:,3]],
                    ), axis = 1)

    return regressor.predict(test_case)
