import pickle

def predict(features):
    # Load the model using pickle
    with open('./lr_cr_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Predict the result based on input features
    res = model.predict(features)
    
    # Print the result
    print(res)

# Predict with sample input
predict([[2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 4]])
