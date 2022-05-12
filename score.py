import os
import logging
import json
import numpy
import pickle
from azure.storage.blob import BlobServiceClient

def init():
    
    blob_connection_string = "DefaultEndpointsProtocol=https;AccountName=mywsstorage07701d4b68654;AccountKey=XHmrGY5DZ4PmqAoPBdyaAWAeH/G+Ox+Az8sg/4Dr0TM1kjiZXiS51DAscIiuVwuCsQIa0lKAFOOI+AStaNR1KA==;EndpointSuffix=core.windows.net"
    blob_container_name = "blob-demo"

    global blob_service_client
    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    
    global model
    #model = joblib.load('model.pkl')
    

def run(raw_data):
    
    blob_client = blob_service_client.get_blob_client(container=blob_container_name,blob=raw_data['model'])
    blob_data = blob_client.download_blob()
    aa = blob_data.readall()
    model = pickle.loads(aa)

    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    y_pred = model.predict(data)
    return y_pred.tolist()
