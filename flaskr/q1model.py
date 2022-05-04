import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

DF_COLS = ['Action Adventure', 'Fantasy', 'Action', 'Platformer', '3D', 'Shooter',
       'First-Person', 'Sci-Fi', 'Adventure', 'Sports', 'Traditional', 'Sim',
       'Modern', 'Fighting', 'Driving', 'Racing', 'Miscellaneous',
       'Compilation', 'Role-Playing', 'General', 'Japanese-Style',
       'Open-World', 'Console-style RPG', 'Third-Person', 'Arcade', 'Strategy',
       'Turn-Based', 'Historic', '2D', 'Survival', 'Action RPG', 'Real-Time',
       'Horror', 'Tactical', 'Simulation', 'Automobile', 'Tactics',
       "Beat-'Em-Up", 'Rhythm', 'Music', 'Combat', 'Puzzle', 'Point-and-Click',
       "Shoot-'Em-Up", 'Team', 'Management', 'Individual', 'E', 'T', 'E10+',
       'M', 'K-A', 'RP', 'AO']

class Q1Model:

    def __init__(self):
        self.umodel = pickle.load(open('flaskr/q1_user_model.sav', 'rb'))
        self.cmodel = pickle.load(open('flaskr/q1_critic_model.sav', 'rb'))

    def predict(self, genres=[], rating=''):
        df_row = {}

        for i in range(len(DF_COLS)):
            df_row[DF_COLS[i]] = [0]

        for col in df_row:
            if col in genres or col == rating:
                df_row[col][0] = 1

        df = pd.DataFrame(df_row)

        return {
            'user_score': self.umodel.predict(df)[0],
            'critic_score': self.cmodel.predict(df)[0]
        }

if __name__ == '__main__':
    print(Q1Model().predict(genres=['Action Adventure', 'Role-Playing', 'Turn-Based'], rating='E'))