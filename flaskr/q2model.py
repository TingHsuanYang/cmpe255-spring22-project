import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

DF_COLS = ['platform_3ds','platform_dreamcast','platform_ds','platform_game-boy-advance','platform_gamecube','platform_nintendo-64','platform_pc',
'platform_playstation','platform_psp','platform_stadia','platform_switch','platform_wii','platform_xbox',
'genre_2D','genre_3D','genre_Action','genre_Action Adventure','genre_Action RPG','genre_Adventure','genre_Alternative','genre_Arcade','genre_Automobile',
'genre_Baseball','genre_Basketball',"genre_Beat-'Em-Up",'genre_Business / Tycoon','genre_Combat','genre_Compilation','genre_Console-style RPG',
'genre_Driving',
'genre_Fantasy',
'genre_Fighting',
'genre_First-Person',
'genre_Flight',
'genre_Football',
'genre_GT / Street',
'genre_General',
'genre_Golf',
'genre_Historic',
'genre_Horror',
'genre_Ice Hockey',
'genre_Individual',
'genre_Japanese-Style',
'genre_Linear',
'genre_Management',
'genre_Massively Multiplayer',
'genre_Massively Multiplayer Online',
'genre_Matching',
'genre_Miscellaneous',
'genre_Mission-based',
'genre_Modern',
'genre_Music',
'genre_Open-World',
'genre_Party / Minigame',
'genre_Platformer',
'genre_Point-and-Click',
'genre_Puzzle',
'genre_Racing',
'genre_Rally / Offroad',
'genre_Real-Time',
'genre_Rhythm',
'genre_Role-Playing',
'genre_Sci-Fi',
"genre_Shoot-'Em-Up",
'genre_Shooter',
'genre_Sim',
'genre_Simulation',
'genre_Soccer',
'genre_Sports',
'genre_Strategy',
'genre_Survival',
'genre_Tactical',
'genre_Tactics',
'genre_Team',
'genre_Third-Person',
'genre_Top-Down',
'genre_Traditional',
'genre_Turn-Based',
'genre_Virtual',
'genre_Virtual Life',
'genre_Visual Novel',
'genre_Western-Style',
'genre_Wrestling',
'rating_E','rating_E10+','rating_M','rating_T']

GENRE_COLS = ['genre_2D',
'genre_3D',
'genre_Action',
'genre_Action Adventure',
'genre_Action RPG',
'genre_Adventure',
'genre_Alternative',
'genre_Arcade',
'genre_Automobile',
'genre_Baseball',
'genre_Basketball',
"genre_Beat-'Em-Up",
'genre_Business / Tycoon',
'genre_Combat',
'genre_Compilation',
'genre_Console-style RPG',
'genre_Driving',
'genre_Fantasy',
'genre_Fighting',
'genre_First-Person',
'genre_Flight',
'genre_Football',
'genre_GT / Street',
'genre_General',
'genre_Golf',
'genre_Historic',
'genre_Horror',
'genre_Ice Hockey',
'genre_Individual',
'genre_Japanese-Style',
'genre_Linear',
'genre_Management',
'genre_Massively Multiplayer',
'genre_Massively Multiplayer Online',
'genre_Matching',
'genre_Miscellaneous',
'genre_Mission-based',
'genre_Modern',
'genre_Music',
'genre_Open-World',
'genre_Party / Minigame',
'genre_Platformer',
'genre_Point-and-Click',
'genre_Puzzle',
'genre_Racing',
'genre_Rally / Offroad',
'genre_Real-Time',
'genre_Rhythm',
'genre_Role-Playing',
'genre_Sci-Fi',
"genre_Shoot-'Em-Up",
'genre_Shooter',
'genre_Sim',
'genre_Simulation',
'genre_Soccer',
'genre_Sports',
'genre_Strategy',
'genre_Survival',
'genre_Tactical',
'genre_Tactics',
'genre_Team',
'genre_Third-Person',
'genre_Top-Down',
'genre_Traditional',
'genre_Turn-Based',
'genre_Virtual',
'genre_Virtual Life',
'genre_Visual Novel',
'genre_Western-Style',
'genre_Wrestling']

class Q2Model:

    def __init__(self):
        self.rfmodel = pickle.load(open('flaskr/q2_rf_model.sav', 'rb'))

    def predict(self, platform='', type='', rating=''):
        df_row = {}

        platform = 'platform_' + platform
        type = 'type_' + type
        rating = 'rating_' + rating

        rf_pop_genre = ''
        rf_max_score = 0

        for i in range(len(DF_COLS)):
            df_row[DF_COLS[i]] = [0]

        for col in df_row:
            if col == platform or col == type or col == rating:
                df_row[col][0] = 1


        for g in GENRE_COLS:
            df_row[g] = 1
            df = pd.DataFrame(df_row)

            rf_score = self.rfmodel.predict(df)[0]
            if (rf_max_score < rf_score):
                rf_max_score = rf_score
                rf_pop_genre = g

            df_row[g] = 0


        return {
            'RandomForestRegressor user_score': float(rf_max_score),
            'RandomForestRegressor recommended genre': str(rf_pop_genre)
        }


if __name__ == '__main__':
    print(Q2Model().predict(platform='platform_gamecube', type='type_singleplayer', rating='rating_T'))
