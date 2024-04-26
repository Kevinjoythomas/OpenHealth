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
   if(request.method=="POST"):
      if 'image' in request.files:
        image_file = request.files['image']

   else:
      return render_template("braintumor.html")
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
    return render_template('ChatBot.html',data=cache['chats'])


@app.route('/test')
def tesasfasf():
   return render_template('test1.html')

@app.route('/features',methods=["POST","GET"])
def features():
   return render_template("features.html")
      
if __name__ == '__main__':
  app.run(debug=True) 