import datetime as dt
import yfinance as yf
import pandas as pd
import sklearn
from sklearn import linear_model
import numpy as np
import pickle

stats = yf.Ticker("FB")
stats.history(period="3mo", reset_index=True, inplace=True)
data = stats.history(period="3mo")

columns = ['Date', 'Open', 'Close', 'Low', 'High', 'Volume']

dates = [day for day in data.index]

data_dict = {
    "Date": dates,
}
for col in columns[1:]:
    temp_list = []
    for val in data[col]:
        temp_list.append(val)

    data_dict[col] = temp_list


df = pd.DataFrame(data_dict)
# print(df)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(dt.datetime.toordinal)

to_predict = "Close"

X = np.array(df.drop([to_predict], 1))
y = np.array(df[to_predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# saving model
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# predictions with test data
predictions = model.predict(np.array(x_test))
for x in range(len(predictions)):
    print(f"Predicted: {predictions[x]}, Real: {y_test[x]}")
    # print(f"Predicted: {dt.datetime.fromordinal(int(predictions[x]))}, Real: {dt.datetime.fromordinal(y_test[x])}")


print("-" * 40)

# prediction with custom data
new_data = {
    "Date": [pd.to_datetime("2022-07-07")],
    'Open': [169.449997],
    'Low': [167.779999],
    'High': [172.72000],
    'Volume:': [23888905]

}  # results for this data is 172.190002

nowy = pd.DataFrame(new_data)

nowy['Date'] = pd.to_datetime(nowy['Date'])
nowy['Date'] = nowy['Date'].map(dt.datetime.toordinal)

predictions = model.predict(np.array(nowy))
for x in range(len(predictions)):
    print(f"Predicted: {predictions[x]}")
