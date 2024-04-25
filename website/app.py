# from flask_cors import CORS
from flask import Flask,render_template


app = Flask(__name__)
# cors = CORS(app,resources={r'/*':{'origin':'*'}})ṇ


@app.route('/')
def test():
    return render_template('index.html')

@app.route('/login')
def log():
    return render_template('login.html')

if __name__ == '__main__':
  app.run(debug=True)
