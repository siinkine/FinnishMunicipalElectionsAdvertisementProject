from flask import Flask, render_template, url_for, request, redirect, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import io
import random

from webapp import utils
import json

#Kuvien piirtäminen aiheuttaa erroria ilman tätä.
plt.switch_backend('agg')

#Datan esikäsittelyä ja muuttujat mm. kuntanumero dropdown valikkoon.
df = pd.read_csv('Data_12_17_21_cleaned.csv')
df = utils.rename_to_english(df)
municipal_names = df['MunicipalName'].unique()



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/' , methods=['GET', 'POST'])

#Tämä funktio käynnistyy, kun 'analyze' nappia painetaan. Ottaa lomakkeiden tiedot talteen ja ajaa analyysifuntkiot
#napatuilla tieodilla. Avaa uuden sivun, jossa grafiikat näytetään.

def to_test():
    if request.method == 'POST':    
        municipal = request.form['Municipal']
        budget = request.form['budget']
        age = request.form['age']
        gender = request.form['sukupuoli']

        json_data, path_im1, path_im2 = perform_master_analysis(municipal, budget, age, gender)

        return render_template('output.html', name = municipal, budget=budget, im1=path_im1, im2=path_im2,  json_data=json_data)
    else:
        return render_template('index2.html', municipal_names = municipal_names)

@app.route('/overall.html')
def overallstatistics():
    return render_template('overall.html')


def perform_master_analysis(municipal, budget, age, gender):
  #  import pdb; pdb.set_trace()
    #datatypes sanity check:
    df = pd.read_csv('Data_12_17_21_cleaned.csv')
    df = utils.rename_to_english(df)
    municipal = municipal
    budget = int(budget)
    age = int(age) #not used
    gender = gender #not used

    #Budget allocation:
    X_chosen, x_vars, df_restric, municipal, path_im1  = utils.budget_allocation(df, municipal, plot=True)
    
    #Univariate regression estimate how many votes:
    y_pred, y_pred_med, path_im2 =  utils.comp_regression(df, municipal,budget, x_name = 'CampaingTotalCosts', y_name = 'TotalVotes', plot=True)
    
    #Multivatiate regression analysis:
    y_votes, y_votes_median, score = utils.multi_var_regression(df, municipal, budget, x_vars, yname = 'TotalVotes')
    #classification:
    y_budget, y_chosen, CM, score = utils.classify_chosen_not(df, municipal, budget)

    #Wrap data to dictionary:
    dct_res ={ 'y_vote_univar': list(y_pred),  'y_vote_med_univar': list(y_pred_med),
                'y_votes_multivar': list(y_votes),  'y_votes_med_multivar': list(y_votes_median),
                'chosen_bud': list(y_budget), 'chosen_med':list(y_chosen)} 
    json_data = json.dumps(dct_res) 

    #plt.plot(data['Sukupuoli'][data['Kuntanro']==user])
    #plt.savefig('static/images/new_plot.png')
    return json_data, path_im1, path_im2


#Käynnistää ohjelman :)

if __name__ == "__main__":
    app.run()
