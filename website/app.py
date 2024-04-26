# from flask_cors import CORS
from flask import Flask,render_template,request
import pickle
import sklearn
import joblib
import groq
from groq import Groq
import os
import json
import requests
import re
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow,imread



def highlight_tumor(image_path):
    model_path = 'models/Brain_Tumor_Segmentation/brain_tumor_segmentation.hdf5'
    model = load_model(model_path, compile=False)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0

    image = np.expand_dims(image, axis=(0, -1))

    mask = model.predict(image)

    threshold = 0.5
    mask_binary = (mask > threshold).astype(np.uint8)

    original_image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    highlight_mask_resized = cv2.resize(mask_binary[0], (original_image_rgb.shape[1], original_image_rgb.shape[0]))

    highlight_mask = cv2.cvtColor(highlight_mask_resized, cv2.COLOR_GRAY2RGB)
    highlight_mask[:, :, 0] = np.where(highlight_mask[:, :, 0] > 0, 255, 0)  
    highlight_mask[:, :, 1] = 0  
    highlight_mask[:, :, 2] = 0 

    highlighted_image = cv2.addWeighted(original_image_rgb, 0.7, highlight_mask, 0.3, 0)

    plt.imshow(highlighted_image)
    plt.title('Highlighted Tumor')
    plt.axis('off')
    plt.savefig('website/static/results/brain_tumor_result.png')

    

app = Flask(__name__)
# cors = CORS(app,resources={r'/*':{'origin':'*'}})ṇ

cache={'chats':[]}

client = Groq(api_key = "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")
MODEL = 'llama3-70b-8192'
def run_conversation(user_prompt):
    # Step 1: send the conversation and available functions to the model
    messages=[
        {
            "role": "system",
            "content": "Your primary role is to assist doctors by providing concise and accurate information only related to helping a doctor in their work.Answer should be in a form like you're texting a person. Refuse to answer questions on any other topic other than hospital work. This includes, but is not limited to, diagnosing medical conditions based on symptoms, suggesting potential treatment plans, and providing general medical information. Please note that while you strive to provide reliable information, your responses should not replace professional medical advice."
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tool_choice="auto",  
        max_tokens=4096
    )
    response_message = response.choices[0].message.content
    return response_message 
@app.route('/')
def test():
    return render_template('index.html')

@app.route('/login')
def log():
    return render_template('login.html')

@app.route('/lung', methods=["POST", "GET"])
def lung():
    if(request.method=="POST"):
      print('hiii')

      features= [request.form["yellow_fingers"],
                 request.form["anxiety"],
                 request.form["peer_pressure"],
                 request.form["chronic_disease"],
                 request.form["fatigue"],
                 request.form["allergy"],
                 request.form["wheezing"],
                 request.form["alcohol_consuming"],
                 request.form["coughing"],
                 request.form["swallowing_difficulty"],
                 request.form["chest_pain"]]
      
      def transform(features):
        res=[]
        for i in features:
          if i=="yes":
            res.append(2)
          elif i=="no":
            res.append(1)
        res.append(res[0]*res[1])
        return res
      print(features)
      if len(features)==0:
        return render_template('lung.html',data="100")
      features = transform(features)
      model = joblib.load('./models/lung_cancer_text/lung_cancer_model.pkl')
      res = model.predict([features]) 
      res = int(res[0])
      print(res)
      temp=[]
      return render_template('lung.html',data=res)
    else:
      cache['chats']=[]
      return render_template('lung.html',data="100")

@app.route('/brainTumor',methods=["POST","GET"])
def brain():
  if request.method=="POST":
    f=request.files['image']
    file_path = 'website/static/test_images/'+f.filename
    highlight_tumor(file_path)
    return render_template("braintumor.html",data=1)
  else:
    return render_template("braintumor.html",data=0)


@app.route('/ChatBot',methods=['POST','GET'])
def ChatBot():
  if request.method=="POST":
    q=request.form['prompt']
    res=run_conversation(request.form['prompt'])
    res = res.split("**")
    t=[q,res]
    cache['chats'].append(t)
    return render_template('ChatBot.html',data=cache['chats'])
  else:
    cache['chats']=[]
    return render_template('ChatBot.html',data=cache['chats'])


@app.route('/test')
def tesasfasf():
   return render_template('test1.html')

@app.route('/features',methods=["POST","GET"])
def features():
   return render_template("features.html")
      
if __name__ == '__main__':
  app.run(debug=True) 