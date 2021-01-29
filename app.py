import uvicorn
from fastapi import FastAPI
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
from pydantic import BaseModel


class MushroomFeatureIN(BaseModel):
    bruises: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_root: str
    population: str
    habitat: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str


# 2. Create the app object
api = FastAPI()
pickle_in = open("model.pkl", "rb")
classifier = pickle.load(pickle_in)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@api.get('/')
def index():
    return {'message': 'Hello, World'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@api.post('/predict')
def predict(data1: MushroomFeatureIN):
    data = data1.dict()
    data = pd.DataFrame(data, index=[0])
    le = LabelEncoder()
    data[['bruises', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_root', 'habitat', 'population',
          'stalk_surface_above_ring', 'stalk_surface_below_ring']] = data[['bruises', 'gill_spacing', 'gill_size',
                                                                           'gill_color', 'stalk_root', 'habitat',
                                                                           'population', 'stalk_surface_above_ring',
                                                                           'stalk_surface_below_ring']].apply(
                                                                                                le.fit_transform)
    data['stalk_surface'] = data['stalk_surface_above_ring'] + data['stalk_surface_below_ring']
    temp = [data['bruises'], data['gill_spacing'], data['gill_size'],
            data['gill_color'], data['stalk_root'], data['habitat'],
            data['population'], data['stalk_surface']]
    temp = np.array(temp)
    temp = temp.reshape(-1, 8)
    prediction = classifier.predict(temp)

    if prediction[0] > 0.5:
        prediction = "Edible Mushroom"
    else:
        prediction = "Poisonous Mushroom"
    return {
        'prediction': prediction
    }


# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(api, host='127.0.0.1', port=8000)
