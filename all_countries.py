import numpy as np, pandas as pd
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_models import LinearRegression

#Grabbing the data:
cases = pd.read_csv("/kaggle/input/time_series_19-covid-Confirmed.csv")
deaths = pd.read_csv("/kaggle/input/time_series_19-covid-Deaths.csv")
recovered = pd.read_csv("/kaggle/input/time_series_19-covid-Confirmed.csv")
countries = [i[0] for i in list(cases.groupby("Country/Region")["Country/Region"])]

#period to be predicted:
start = datetime.datetime.strptime("03-05-20", "%m-%d-%y")
end = datetime.datetime.strptime("06-07-20", "%m-%d-%y")
date = start
date_list = []
final_prediction  = {}
date_comparison = []
while(date.timestamp()<=end.timestamp()):
    date += datetime.timedelta(days = 1)
    date_comparison.append(date)
    date_list.append(date.timestamp())

#funcitons needed:
def generatePoints(predictions):
    # generate x and y from predictions array
    points = []
    x = 0
    for i in predictions:
        points.append({ "x": x ,"y": round(i) });
        x += 1

    return points
def swap(x, format="%m/%d/%y"):
    """Generates timestraps from dates"""
    return datetime.datetime.strptime(x, format).timestamp()
def country_predictions(country, data):
        """generates predictions from the chosen data and country"""
        
        i = 4
        try: 
            while( data[data["Country/Region"] == country][data.columns[i:i+1]].apply(sum).values[0]<=0):
                i+=1
        except: i = 4
        if( i>4):
            data = data[data["Country/Region"] == country][data.columns[i-1:]].apply(sum)
        else:
            data = data[data["Country/Region"] == country][data.columns[i:]].apply(sum)
	data = data[data["Country/Region"] == country][data.columns[4:]].apply(sum)
        x = data.index
        y = data.values
        x_stmps= pd.Series(x).apply(swap)
        poly = PolynomialFeatures(degree = 4)
        X_Poly = poly.fit_transform(np.array(x_stmps).reshape(len(x_stmps), 1)[45:])
        poly.fit(X_Poly, y[45:])
        #Fitting data:
        model_linear = LinearRegression()
        model_linear.fit(X_Poly, y[45:])
        predictions = model_linear.predict(poly.fit_transform(np.array(date_list).reshape(len(date_list), 1)))
        return generatePoints(predictions)

#Extracting data for each country:
#Data for infections of each country:
country_data = {}

for country in countries:
    country_data[country] = {'cases': country_predictions(country, cases), 'deaths': country_predictions(country, deaths), 'recovered': country_predictions(country, recovered)}
