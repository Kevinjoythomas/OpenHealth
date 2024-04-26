# from flask_cors import CORS
from flask import Flask,render_template,request


app = Flask(__name__)
# cors = CORS(app,resources={r'/*':{'origin':'*'}})ṇ


@app.route('/')
def test():
    return render_template('index.html')

@app.route('/login')
def log():
    return render_template('login.html')

@app.route('/lung', methods=["POST", "GET"])
def lung():
    if(request.method=="POST"):

      features= [request.form["yellow_fingers"],request.form["anxiety"],request.form["peer_pressure"],request.form["chronic_disease"],
                 request.form["fatigue"],
                 request.form["allergy"],
                 request.form["wheezing"],
                 request.form["alcohol_consuming"],
                 request.form["coughing"],
                 request.form["swallowing_difficulty"],
                 request.form["chest_pain"]]
      return render_template('lung.html')
    else:
      return render_template('lung.html')

   

