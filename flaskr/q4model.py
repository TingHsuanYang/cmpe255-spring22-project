import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

DF_COLS = ['platform_3ds', 'platform_dreamcast', 'platform_ds',
           'platform_game-boy-advance', 'platform_gamecube',
           'platform_nintendo-64', 'platform_pc', 'platform_playstation',
           'platform_psp', 'platform_stadia', 'platform_switch', 'platform_wii',
           'platform_xbox', 'genre_2D', 'genre_3D', 'genre_Action',
           'genre_Action Adventure', 'genre_Action RPG', 'genre_Adventure',
           'genre_Alternative', 'genre_Arcade', 'genre_Automobile',
           'genre_Basketball', 'genre_Beat-\'Em-Up', 'genre_Combat',
           'genre_Compilation', 'genre_Console-style RPG', 'genre_Driving',
           'genre_Fantasy', 'genre_Fighting', 'genre_First-Person',
           'genre_Football', 'genre_GT / Street', 'genre_General',
           'genre_Historic', 'genre_Horror', 'genre_Individual',
           'genre_Japanese-Style', 'genre_Linear', 'genre_Management',
           'genre_Miscellaneous', 'genre_Modern', 'genre_Music',
           'genre_Open-World', 'genre_Platformer', 'genre_Puzzle', 'genre_Racing',
           'genre_Real-Time', 'genre_Rhythm', 'genre_Role-Playing', 'genre_Sci-Fi',
           'genre_Shoot-\'Em-Up', 'genre_Shooter', 'genre_Sim', 'genre_Simulation',
           'genre_Soccer', 'genre_Sports', 'genre_Strategy', 'genre_Survival',
           'genre_Tactical', 'genre_Tactics', 'genre_Team', 'genre_Third-Person',
           'genre_Traditional', 'genre_Turn-Based', 'genre_Western-Style',
           'type_multiplayer', 'type_singleplayer', 'rating_E', 'rating_E10+',
           'rating_M', 'rating_T']

PLATFORM_COLS = ['platform_3ds',
                'platform_dreamcast',
                'platform_ds',
                'platform_game-boy-advance',
                'platform_gamecube',
                'platform_nintendo-64',
                'platform_pc',
                'platform_playstation',
                'platform_psp',
                'platform_stadia',
                'platform_switch',
                'platform_wii',
                'platform_xbox']

class Q4Model:

    def __init__(self):
        self.seqmodel = pickle.load(open('flaskr/q4_seq_model.sav', 'rb'))
        self.rfrmodel = pickle.load(open('flaskr/q4_rfr_model.sav', 'rb'))
        # self.seqmodel = pickle.load(open('q4_seq_model.sav', 'rb'))
        # self.rfrmodel = pickle.load(open('q4_rfr_model.sav', 'rb'))

    def predict(self, genres=[], type='', rating=''):
        df_row = {}

        genres = ["genre_" + s for s in genres]
        type = 'type_' + type
        rating = 'rating_' + rating


        seq_pop_platform = ''
        rfr_pop_platform = ''
        seq_max_score = 0
        rfr_max_score = 0

        for i in range(len(DF_COLS)):
            df_row[DF_COLS[i]] = [0]

        for col in df_row:
            if col in genres or col == rating:
                df_row[col][0] = 1


        for p in PLATFORM_COLS:
            df_row[p] = 1
            df = pd.DataFrame(df_row)
            
            seq_score = self.seqmodel.predict(df)[0]
            if (seq_max_score < seq_score):
                seq_max_score = seq_score
                seq_pop_platform = p

            rfr_score = self.rfrmodel.predict(df)[0]
            if (rfr_max_score < rfr_score):
                rfr_max_score = rfr_score
                rfr_pop_platform = p

            df_row[p] = 0


        return {
            'Sequential user_score': float(seq_max_score),
            'Sequential predicted platform': str(seq_pop_platform),
            'RandomForestRegressor user_score': float(rfr_max_score),
            'RandomForestRegressor predicted platform': str(rfr_pop_platform)
        }


if __name__ == '__main__':
    print(Q4Model().predict(genres=['genre_3D', 'genre_Shooter', 'genre_Horror'], type='type_singleplayer', rating='rating_E'))
