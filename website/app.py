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

@app.route('/test', methods=["POST", "GET"])
def test1():
    if(request.method=="POST"):
      print('abcd')
      name = request.form['name']
      email = request.form['email']
      return render_template('test1.html', name=name,email=email)
    else:
      print('abc')
      return render_template('test.html')

if __name__ == '__main__':
  app.run(debug=True)
