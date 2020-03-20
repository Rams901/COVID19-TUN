import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Importing data:
df = pd.read_csv("csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")

#Initializing the x/y variables:
x = df.columns[4:]
y = df[df["Country/Region"] == "Tunisia"].drop(df.columns[:4], axis = 1)

#Function to change all the x datetimes into timestamps:
def swap(x, format="%m/%d/%y"):
    return datetime.datetime.strptime(x, format).timestamp()
x_stmps= pd.Series(x).apply(swap)

#Initializing the period to be predicted:
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


from sklearn.preprocessing import PolynomialFeatures
#Initializing and transforming the data:
poly = PolynomialFeatures(degree = 4)
X_Poly = poly.fit_transform(np.array(x_stmps).reshape(58, 1)[45:])
poly.fit(X_Poly, y[45:])
#Fitting data:
model_linear = LinearRegression()
model_linear.fit(X_Poly, y[45:])
#Testing & Visualization:
plt.figure(figsize=(7, 9))

ax = plt.subplot(111)

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 

#plotting the predicted data:  
predictions = poly.fit_transform(np.array(date_list).reshape(len(date_list), 1)))
plt.plot(  model_linear.predict(predictions[10:], lw = 3, color = "red", alpha = 0.6)
plt.text(86, 1600, "Infections", fontsize=14, color="red", alpha = 0.6) 
plt.yticks(fontsize = 14)
ax.set_xticks([3, 27 , 58, 88 ])
ax.set_xticklabels(['Mars', 'April', 'May', 'June', 'July'])
plt.xticks(fontsize = 14)
plt.title("Predictions For The Number Of infected Cases In Tunisia (Mars - June)", fontsize = 16)
