from flask_cors import CORS
from flask import Flask,render_template


app = Flask(__name__)
cors = CORS(app,resources={r'/*':{'origin':'*'}})


@app.route('/')
def test():
    return render_template('index.html')


if __name__ == '__main__':
  app.run(debug=True)
