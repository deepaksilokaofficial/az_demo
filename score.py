import os
import json
import numpy
import joblib


def init():
    
    global model
    model = joblib.load('model.pkl')
    

def run(raw_data):
    
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    y_pred = model.predict(data)
    return y_pred.tolist()
