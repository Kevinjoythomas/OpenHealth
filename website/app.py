# from flask_cors import CORS
from flask import Flask,render_template,request,redirect
import pickle
import sklearn
import joblib
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("C:\\Users\\karthikeya\\Dropbox\\PC\\Downloads\\openhealth-25698-firebase-adminsdk-mmnxo-b16e2f7b80.json")

firebase_admin.initialize_app(cred)

app = Flask(__name__)
# cors = CORS(app,resources={r'/*':{'origin':'*'}})ṇ

db = firestore.client()
session=0

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

      return render_template('lung.html',data=res)
    else:
      return render_template('lung.html',data="100")

@app.route('/brainTumor',methods=["POST","GET"])
def brain():
   if(request.method=="POST"):
      if 'image' in request.files:
        image_file = request.files['image']

   else:
      return render_template("braintumor.html")

@app.route('/test')
def tesasfasf():
   return render_template('test1.html')

@app.route('/features',methods=["POST","GET"])
def features():
   return render_template("features.html")


@app.route('/login',methods=["POST","GET"])
def login():
   if(request.method=="post"):
      email=request.form["email"]
      password=request.form["password"]
      user = db.collection('doctors').where('email', '=', email).limit(1).get()
      if (user== None):
         return redirect("/login")
      else:
         if(user.password==password):
            session=1
            return redirect("/features")
         else:
            return redirect("/login")
         
   else:
      return render_template("login.html")
   
@app.route('/signup',methods=["POST","GET"])
def signup():
   if(request.method=="post"):
      email=request.form["email1"]
      password=request.form["password1"]
      name=request.form['name1']
      specialisation=request.form['specialisation']
      user = db.collection('doctors').where('email', '=', email).limit(1).get()
      if(user!=None):
         return redirect('/singup')
      else:
         data= {
        'name': name,
        'email': email,  # Convert to int if needed
        'password': password,
        'specialisation':specialisation
               }
         doc=db.collection('doctors').add(data)
         print("signin successfully")
   else:
      return render_template("login.html")


   
         



@app.route('/notifications',methods=["GET","POST"])
def notification():
   docs = db.collection('notifications').get()
   if(docs==None):
      return render_template('')
   else:
      return redirect()
      
         
      
if __name__ == '__main__':
  app.run(debug=True) 