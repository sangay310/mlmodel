from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open("mymodel3 (2).pkl",'rb'))
cols = ["Gender","Age","Height(meter)","Weight","Frequent_Intake_of_Caloric_Food","Vegetable_in_meal","No_of_meals/day","High_Caloric_Food","Smoke","Consumption_of_Water(L)","Monitaring_of_calories","Physical_Activity(hr)","Time_spend_using_tech(hr)","Frequency_of_Alcohol","Mode_Of_Transportation"]



@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    Gender = request.form['Gender']
    Age = request.form['Age']
    Height = request.form['Height']
    Weight = request.form['Weight']
    Frequent_Intake_of_Caloric_Food = request.form['Frequent_Intake_of_Caloric_Food']
    Vegetable_in_meal = request.form['Vegetable_in_meal']
    No_of_meals_day = request.form['No_of_meals/day']
    High_Caloric_Food = request.form['High_Caloric_Food']
    Smoke = request.form['Smoke']
    Consumption_of_Water = request.form['Consumption_of_Water'] # type: ignore
    Monitaring_of_calories = request.form['Monitaring_of_calories']
    Physical_Activity = request.form['Physical_Activity(hr)']
    Time_spend_using_tech = request.form['Time_spend_using_tech(hr)']
    Frequency_of_Alcohol = request.form['Frequency_of_Alcohol']
    Mode_Of_Transportation = request.form['Mode_Of_Transportation']

        # Create a numpy array from the extracted form data
    arr = np.array([[Gender, Age, Height, Weight, 
                            Frequent_Intake_of_Caloric_Food, Vegetable_in_meal, No_of_meals_day,
                            High_Caloric_Food, Smoke, Consumption_of_Water, Monitaring_of_calories, # type: ignore
                            Physical_Activity, Time_spend_using_tech, Frequency_of_Alcohol,
                            Mode_Of_Transportation]])
    arr2 = pd.DataFrame(arr, columns=cols)
    pred = model.predict(arr2)
    return render_template('result.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)
