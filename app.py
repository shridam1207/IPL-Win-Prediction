from flask import Flask
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV 
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
from flask import render_template
from flask import request
import pickle


model = pickle.load(open('model.pkl', 'rb'))

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")


app = Flask(__name__)
ground_dict = {'Sunrisers Hyderabad' : 'Hyderabad','Mumbai Indians':'Mumbai', 'Gujarat Lions':'Rajkot',
       'Rising Pune Supergiant':'Pune', 'Royal Challengers Bangalore':'Bangalore',
       'Kolkata Knight Riders':'Kolkata', 'Delhi Daredevils':'Delhi', 'Kings XI Punjab':'Dharamsala',
       'Chennai Super Kings':'Chennai', 'Rajasthan Royals':'Jaipur', 'Deccan Chargers':'Hyderabad',
       'Kochi Tuskers Kerala':'Kochi', 'Pune Warriors':'Pune', 'Rising Pune Supergiants':'Pune'}


international_grounds = ['Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
       'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Abu Dhabi', 'Sharjah']


teams = {'Sunrisers Hyderabad':0, 'Mumbai Indians':1, 'Gujarat Lions':2,
       'Rising Pune Supergiant':3, 'Royal Challengers Bangalore':4,
       'Kolkata Knight Riders':5, 'Delhi Daredevils':6, 'Kings XI Punjab':7,
       'Chennai Super Kings':8, 'Rajasthan Royals':9, 'Deccan Chargers':10,
       'Kochi Tuskers Kerala':11, 'Pune Warriors':12, 'Rising Pune Supergiants':13}

def preprocess_data(r):

# def rem_target(r):
    if r.loc[0]['score_target'] == -1:
        r['remaining_target'] = -1

    else:
         r['remaining_target'] = r.iloc[0]['score_target'] - r.iloc[0]['innings_score']
    

# def run_rate(r):
    r['run_rate'] = r.iloc[0]['innings_score'] / r.iloc[0]['over']


# def req_run_rate(r):
    if r.iloc[0]['remaining_target'] == -1:
        r['required_run_rate'] =-1
    elif r.iloc[0]['over'] == 20:
        r['required_run_rate'] = 100
    else:
        r['required_run_rate'] =  r.iloc[0]['remaining_target'] / (20-r.iloc[0]['over'] )


    r['runrate_diff'] = r.iloc[0]['required_run_rate'] - r.iloc[0]['run_rate']
        

# def city_type(r):
    if ground_dict[r.iloc[0]['team1']] == r.iloc[0]['city']:
        r['city_type']= 0
    elif ground_dict[r.iloc[0]['team2']] == r.iloc[0]['city']:
        r['city_type']= 1
    elif r.iloc[0]['city'] in international_grounds:
        r['city_type']= 2
    else:
        r['city_type']= 3


# def team_encoder(r):
    r['team1_cat'] = teams[r.iloc[0]['team1']]
    r['team2_cat'] = teams[r.iloc[0]['team2']]


    r.drop(['team1','team2','city'], axis=1, inplace = True)


@app.route('/')
def index():
    return render_template("index.html")

# prediction function
def ValuePredictor(to_predict_list):



    columns2 = ['inning', 'over', 'total_runs', 'player_dismissed',
       'innings_wickets', 'innings_score', 'score_target', 'city','team1','team2']
    
    x = dict(zip(columns2,to_predict_list))

    y = pd.DataFrame(x, index = [0])
    team1 = y.iloc[0]['team1']
    team2 = y.iloc[0]['team2']
    for i in ['inning', 'over', 'total_runs', 'player_dismissed', 'innings_wickets',
       'innings_score', 'score_target',]:
       y[i] = y[i].astype('int')

    preprocess_data(y)

    # clf = RandomForestClassifier().fit(X_train, y_train) 
    vel = model.predict(y)

    if vel == 1:
        return team1
    else:
        return team2


    return vel

 


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        #to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        return render_template("index.html", prediction_text = "The predicted Winner of the match is {}".format(result))



if __name__ == "__main__":
    app.run(debug=True)