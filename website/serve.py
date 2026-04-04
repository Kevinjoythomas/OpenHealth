"""Minimal static server for the OpenHealth HTML frontend."""
from flask import Flask, render_template, redirect

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/login")
def login():
    return render_template("login.html")

@app.get("/ChatBot")
def chatbot():
    return render_template("main_chatbot.html")

if __name__ == "__main__":
    app.run(port=3000, debug=True)
