# from flask_cors import CORS
from flask import Flask,render_template,request,redirect
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
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow,imread
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from langchain_ollama import OllamaLLM
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory




# import speech_recognition 
# import pyttsx3

CHROMA_PATH = "./chroma"
PROMPT_TEMPLATE = """
You are a medical professional. Answer the question like a doctor would. It should consist of paragraph and conversational aspect rather than just a summary. Answer the asked question to the point briefly and dont make up stuff. Answer in a professional tone:

{context}

---

Answer the question based on your knowledge and using the above context for help: {question}
"""

cred = credentials.Certificate('./static/secretKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
gatedModel = OllamaLLM(model="llama3")
embedding_function = OllamaEmbeddings(model="nomic-embed-text")
chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
llama_model = OllamaLLM(model="llama3.2")


# recognizer = speech_recognition.Recognizer()

# def speech_to_bot():
#       with speech_recognition.Microphone() as mic:
#               print("listening")
#               recognizer.adjust_for_ambient_noise(mic,duration=0.2)
#               print("listening")
                
#               audio = recognizer.listen(mic)
#               print("listening")
              
#               text = recognizer.recognize_google(audio)

#               text = text.lower()

#               print(text)
#               return text

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
    plt.close()
    

app = Flask(__name__)
# cors = CORS(app,resources={r'/*':{'origin':'*'}})ṇ

cache={'chats':[]}

def get_session_history(session:str)->BaseChatMessageHistory:
    if session_id not in sessions:
        sessions[session_id] = ChatMessageHistory()
    return sessions[session_id]
  
def run_conversation(user_prompt:str):
    try:
        print("Received user prompt:", user_prompt, flush=True)

        gatePrompt = f"<|start_header_id|>system<|end_header_id|>I will now give you a question. This question should only be related to medical queries or advice. If it is related to medical queries or advice, then reply with 'True' and nothing else, no explanation, nothing, just 'True'. If it's not related to medical info, then just say 'False' and nothing else, no explanation, nothing, just 'False'. Just reply with either True or False and nothing else.<|eot_id|><|start_header_id|>user<|end_header_id|> This is the question: {user_prompt}<|eot_id|>"
        gateResult = gatedModel.invoke(gatePrompt)
        print("Gate Result:", gateResult, flush=True)

        if gateResult.strip().lower() == "false":
            print("Query not related to medical field.", flush=True)
            return "This query is not related to the medical field. Please ask related queries."

        # print("Chroma DB initialized successfully.", len(chroma_db.get()), flush=True)

        # try:
        #     results = chroma_db.similarity_search(user_prompt, k=1)
        # except MemoryError:
        #     print("MemoryError occurred while performing similarity search.")
            
        # print("Similarity search results:", results, flush=True)

        # context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        # prompt = prompt_template.format(context=context_text, question=user_prompt)
        # print("Prompt created:", prompt, flush=True)

        # response = llama_model.invoke(prompt)
        # print("Model Response:", response, flush=True)
        retriever = chroma_db.as_retriever()
        
        contextualize_q_system_prompt=(
                    "Given a chat history and the latest user question"
                    "which might reference context in the chat history, "
                    "formulate a standalone question which can be understood "
                    "without the chat history. Do NOT answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llama_model,retriever,contextualize_q_prompt)
        system_prompt = (
            "You are a highly experienced medical professional communicating with a patient via text. Your role is to provide clear, concise, and empathetic advice based on the patient's concerns, as you cannot perform physical examinations remotely. Maintain a professional and approachable tone, similar to a doctor conversing with a patient in a hospital. Engage in a conversational manner rather than offering dry summaries."
            "\n\n"
            "Your primary goals are:"
            "1. To provide accurate medical advice or guidance based on the given context."
            "2. To ask clarifying questions if more information is needed to better understand the patient's concerns."
            "3. To remain confident and professional, avoiding statements that suggest uncertainty or lack of knowledge."
            "\n\n"
            "{context}"
            "\n\n"
            "Use the information above and your medical expertise to craft responses tailored to the patient's needs. Ensure the advice is consice, actionable, empathetic, and relevant to the context provided and NEVER give the hint that you dont know a particular topic or ask them to meet anyone else for help."
        )

        
        qa_prompt = ChatPromptTemplate.from_messages(
          [
              ("system",system_prompt),
              MessagesPlaceholder("chat_history"),
              ("human","{input}")
          ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llama_model,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key = "chat_history",
            output_messages_key = "answer"
        )
        response  = conversational_rag_chain.invoke(
            {"input":user_prompt},
            config={
                "configurable":{"session_id":session_id}
            }
        )
        return response["answer"]

    except Exception as e:
        import traceback
        print(f"Error in query: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return f"Error processing request: {e}"
    
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
      
      return render_template('lung.html',data="100")

@app.route('/brainTumor',methods=["POST","GET"])
def brain():
   if(request.method=="POST"):
      if 'image' in request.files:
        print(1)
        image_file = request.files['image']
        image_path = 'website/static/test_images/'+image_file.filename
        highlight_tumor(image_path)
      print(2)
      return render_template("braintumor.html",data=1)
   
   else:
      return render_template("braintumor.html")

@app.route('/test')
def tesasfasf():
   return render_template('test1.html')

@app.route('/features',methods=["POST","GET"])
def features():
   return render_template("features.html")
      

@app.route('/ChatBot',methods=["POST","GET"])
def ChatBot():

  if(request.method=="POST"):
    
    q=request.form['prompt']
    res=""
    # if len(q)==0:
      # q=speech_to_bot()
    q = str(q)
    res = run_conversation(q)

    temp=[]
    temp.append(q)
    res = res.split("**")
    temp=[q,res]
    cache['chats'].append(temp)

    return render_template('ChatBot.html',data=cache['chats'])
  else:
    cache['chats']=[]
    return render_template('ChatBot.html',data=cache['chats'])


@app.route('/diabetes')
def diabetes():
  #  pregnencies = request.form['pregnancies']
  #  glucose = request.form['glucose']
  #  blood_pressure = request.form['blood_pressure']
  #  skin_thickness = request.form['skin_thickness']
  #  insulin = request.form['insulin']
  #  bmi = request.form['bmi']
  #  diabetes_pedigree_function = request.form['diabetes_pedigree_function']
  #  age = request.form['ages']

   return render_template('diabetes.html')

@app.route('/login',methods=["POST","GET"])
def login():
   if(request.method=="POST"):
      print(1)
      email2=request.form["email"]
      print(email2)
      password=request.form["password"]
      print(password)
      user = db.collection('doctors').where('email', '==', email2).get()
      print(type(user))
      if (not user):
         print(user)
         print(2)
         return redirect("/login")
      else:
        print(3)
        user=user[0]
        user=user._data
        if(user['password']==password):
            print(4)
            return redirect("/features")

        else:
          print(5)
          return redirect("/login")
         
   else:
      print(6)
      return render_template("login.html")
   
@app.route('/signup',methods=["POST","GET"])
def signup():
  
   if(request.method=="POST"):
      print("post")
      email=request.form["email1"]
      print(email)
      password=request.form["password1"]
      print(password)
      name=request.form['name1']
      print(name)
      docs = db.collection('doctors').get()
      for doc in docs:
          print(doc.id, doc.to_dict())
      user = db.collection('doctors').where('email', '==', email).limit(1).get()
      if(len(user)!=0):
         return redirect('/signup')
      else:
         print("saving data")
        
         data= {
        'name': name,
        'email': email,  # Convert to int if needed
        'password': password,
               }
         doc=db.collection('doctors').add(data)
         cache['currentUser']=email
         print("signin successfully")
         return redirect('/features')
   else:
      return render_template("login.html")


   
         



@app.route('/notifications',methods=["GET","POST"])
def notification():
   if(request.method=='POST'):
      name=request.form["name"]
      num=request.form["num"]
      num=int(num)
      num-=1  
      data= {
        'name': name,
        'no of doctors':num
               }
      doc=db.collection('notification').add(data)
      return redirect('/notifications')
    
   else:
    docs = db.collection('notification').get()
    print(len(docs))
    print(docs)
    return render_template("notifications.html",docs=docs)
   
   
@app.route("/accept/<string:name>", methods=["POST", "GET"])
def accept(name):
    # Query the 'notifications' collection for documents with the specified name
    user_ref = db.collection('notification').where('name', '==', name).get()
    
    # Check if any documents were found
    if user_ref:
        for doc in user_ref:
            # Get the current value of the field 'num' from each document
            current_no_of_doctors = doc._data.get('num', 0)
            # Increment the value by 1
            updated_data = {'num': current_no_of_doctors + 1}
            doc.reference.update(updated_data)
            
            # Check if the condition is met to delete the document
            if doc._data.get("no of doctors") == doc._data.get("num"):
                doc.reference.delete()
        
        # After updating and potentially deleting documents, redirect to another route
        return redirect('/notifications')
    else:
        return "No notifications found for the specified name."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,threaded=False)