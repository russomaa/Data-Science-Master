from pymongo import MongoClient
import pandas as pd

client = MongoClient('localhost', 27017)
db = client['Tweets']
collection = db['avengers']
data = pd.DataFrame(list(collection.find({})))
data.to_csv('coordenadas1.csv')
print(data['date'].apply(lambda t: t.strftime('%d-%m-%Y %H:%M')))
