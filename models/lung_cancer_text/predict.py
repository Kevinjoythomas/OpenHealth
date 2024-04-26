import pickle
import sklearn
import joblib

def predict(features):
    model = joblib.load('./models/lung_cancer_text/lung_cancer_model.pkl')
    res = model.predict(features) 
    print(res)

predict([[2,2,2,1,1,2,2,1,2,2,2,4]])