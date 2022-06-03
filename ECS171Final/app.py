from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras as keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense, Embedding, Flatten
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from keras.callbacks import EarlyStopping
import pickle

app = Flask(__name__, template_folder='public', static_folder='static')

model = pickle.load(open("model_adabo.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods =["POST"])
def predict():
  #For rendering results on HTML GUI
  bmi = request.form.get("BMI")
  print(bmi)
  bmi = (int(bmi) - 12.02)/(94.85 - 12.02)
  smoker = request.form.get("smoker")
  alcohol = request.form.get("alcohol")
  stroke = request.form.get("stroke")
  walking = request.form.get("walking")
  sex = request.form.get("sex")
  race = request.form.get("race")
  diabetic = request.form.get("diabetic")
  active = request.form.get("active")
  kidney = request.form.get("kidney")
  asthma = request.form.get("asthma")
  skin = request.form.get("skin")
  age = request.form.get("age")
  health = request.form.get("health")
  ph = request.form.get("ph")
  mh = request.form.get("mh")
  sleep = request.form.get("sleep")

  inputsDict = {"Male":1, "Female":0, "American Indian/Alaskan Native":0, "Asian":0, "Black":0, "Hispanic":0, "White":1, "Other":0, "Poor":0, "Fair":1, "Good":2, "Very Good":3, "Excellent":4, None:0, "smoker":1, "alcohol":1, "stroke":1, "walking":1, "diabetic":1, "active":1, "kidney":1, "asthma":1, "skin":1, "18-24":0, "25-29":1, "30-34":2, "35-39":3, "40-44":4, "45-49":5, "50-54":6, "55-59":7, "60-64":8, "65-69":9, "70-74":10, "75-79":11, "80 or older":12}

  inputs = np.array([bmi, inputsDict[smoker], inputsDict[alcohol], inputsDict[stroke], ph, mh, inputsDict[walking], inputsDict[sex], inputsDict[age], inputsDict[race], inputsDict[diabetic], inputsDict[active], inputsDict[health], sleep, inputsDict[asthma], inputsDict[kidney], inputsDict[skin]])

  inputs = inputs.reshape(1, -1)

  prediction = model.predict(inputs)

  predText = ""
  if prediction == 0:
      predText = "The model predicts you are not at risk of heart disease."
  else:
      predText = "The model predicts you are at risk of heart disease."
  
  return render_template('index.html', prediction_text=predText)

app.run(host='0.0.0.0', port=81)