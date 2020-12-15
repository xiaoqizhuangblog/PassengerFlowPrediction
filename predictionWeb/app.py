from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import math
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

def time_to_period(time):
    if time >= 45:
        return 45
    elif time >= 30:
        return 30
    elif time >= 15:
        return 15
    else:
        return 0

data = pd.read_csv('r66.csv')
study_weekday = pd.read_csv('study_weekday.csv')
study_weekend = pd.read_csv('study_weekend.csv')
holiday_weekday = pd.read_csv('holiday_weekday.csv')
holiday_weekend = pd.read_csv('holiday_weekend.csv')
data = data.drop(data.columns[0],axis = 1)
data["Date"] = pd.to_datetime(data["Date"])
study_weekday["Date"] = pd.to_datetime(study_weekday["Date"])
study_weekend["Date"] = pd.to_datetime(study_weekend["Date"])
holiday_weekday["Date"] = pd.to_datetime(holiday_weekday["Date"])
holiday_weekend["Date"] = pd.to_datetime(holiday_weekend["Date"])


values = data.iloc[:,3:].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':

        year = request.form['year']
        month = request.form['month']
        day = request.form['day']

        match = data[ (data["Date"] == '{}-{}-{}'.format(year,month,day))].iloc[0,:]

        if match.Semester == 1:
            if (match.Weekday >= 0)&(match.Weekday <= 4):
                matched = study_weekday[ (study_weekday["Date"] == '{}-{}-{}'.format(year,month,day))]
                values = study_weekday.iloc[:,4:].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(values)
                model = tf.keras.models.load_model("study_weekday")
                print("study_weekday")
            else:
                matched = study_weekend[ (study_weekend["Date"] == '{}-{}-{}'.format(year,month,day))]
                values = study_weekend.iloc[:,4:].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(values)
                model = tf.keras.models.load_model("study_weekend")
                print("study_weekend")
        else:
            if (match.Weekday >= 0)&(match.Weekday <= 4):
                matched = holiday_weekday[ (holiday_weekday["Date"] == '{}-{}-{}'.format(year,month,day))]
                values = holiday_weekday.iloc[:,4:].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(values)
                model = tf.keras.models.load_model("holiday_weekday")
                print("holiday_weekday")
            else:
                matched = holiday_weekend[ (holiday_weekend["Date"] == '{}-{}-{}'.format(year,month,day))]
                values = holiday_weekend.iloc[:,4:].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(values)
                model = tf.keras.models.load_model("holiday_weekend")
                print("holiday_weekend")

        time_step = 420
        prediction = np.zeros(60)

        pre_data = scaled[ matched.index[0] - time_step  : matched.index[0]  ,:]
        pre = model.predict(np.reshape(pre_data,([1,420,4])))
        matchedScale = scaled[matched.index[0]]
        dayInfo = matchedScale[1:]
        
        for i in range(60):
            prediction[i] = pre
            pre_data_now = np.concatenate((pre, dayInfo.reshape([1,3])), axis=1)
            pre_data_past = pre_data[1:, :]
            pre_data = np.concatenate((pre_data_past,pre_data_now), axis=0)
            pre = model.predict(np.reshape(pre_data,([1,420,4])))

        prediction = np.concatenate((prediction.reshape([60,1]), np.zeros([60,3])), axis=1)
        prediction = scaler.inverse_transform(prediction)
        prediction = prediction[:,0]

        pre_sum = sum(prediction)
        true_sum = sum(matched["Passengers"])
        if abs(pre_sum - true_sum) < 1000:
            goodness = True
        else:
            goodness = False

        out = []
        for x in [str(x) for x in range(6,21)]:
            x = "0" + x if len(x) == 1 else x
            for i in [str(x) for x in range(60)][::15]:
                i = "0" + i if len(i) == 1 else i
                out.append("{}:{}:00".format(x, i))
                
        prediction = [int(i) for i in prediction]
        table = pd.DataFrame({'TimePeriod':out,'PassengerFlow':prediction})
        df_list = table.values.tolist()
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(range(len(matched["Passengers"])), matched["Passengers"])
    ax.plot(prediction)
    
    ax.set_title("Passenger flow on {}-{}-{}".format(year,month,day))
    ax.set_xlabel("Time: 0 = 6:00am, 60 = 21:00pm")
    ax.set_ylabel("Passenger flow")
    ax.legend(["True","Prediction"])
    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    plt.close()

    return render_template('result.html', image=pngImageB64String, year = year, month = month, 
                            day = day, prediction = int(round(pre_sum)), 
                            true_sum = int(round(true_sum)), my_list=df_list, goodness = goodness)


    


@app.route('/prediction')
def prediction():
    return render_template("prediction.html")

@app.route('/model')
def model():
    return render_template("model.html")


if __name__ == '__main__':
	app.run(debug=True)