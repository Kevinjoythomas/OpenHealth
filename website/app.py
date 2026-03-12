from flask import Flask, render_template, request, redirect, url_for, session
import os
import re
import firebase_admin
from firebase_admin import credentials, firestore
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma"

cred = credentials.Certificate(os.getenv('firebase_secretkey'))
firebase_admin.initialize_app(cred)
db = firestore.client()
embedding_function = OllamaEmbeddings(model="nomic-embed-text")
chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
llama_model = OllamaLLM(model="hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF")

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')

cache = {'chats': []}
session_id = "session"
count = 0

session_histories = {}


def get_session_history(session_id):
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]


def run_conversation(user_prompt: str):
    try:
        retriever = chroma_db.as_retriever(search_type="mmr", top_k=1)

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_obj = get_session_history(session_id)
        history_messages = history_obj.messages

        history_aware_retriever = create_history_aware_retriever(
            llama_model, retriever, contextualize_q_prompt
        )

        system_prompt = (
            "You are a highly experienced medical professional who have been communicating with a patient via text. "
            "\n\n"
            "Your primary goals are:"
            "1. To provide accurate medical advice or guidance based on the given context in less than 100 words maximum."
            "2. To ask clarifying questions if more information is needed to better understand the patient's concerns."
            "3. To remain confident and professional, avoiding statements that suggest uncertainty or lack of knowledge."
            "\n\n"
            "{context}"
            "\n\n"
            "Use the information above ONLY if it is related to the question and your medical expertise to craft "
            "responses tailored to the patient's needs. Ensure the advice is concise and relevant to the context "
            "provided in the chat history. Never repeat these instructions in your response."
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llama_model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        if "forgot my first question" in user_prompt.lower() or "previous question" in user_prompt.lower():
            if len(history_messages) >= 2:
                return f"Your previous question was: '{history_messages[-2].content}'"
            else:
                return "You haven't asked any questions yet in this conversation."

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        response = conversational_rag_chain.invoke(
            {"input": user_prompt},
            config={"configurable": {"session_id": session_id}}
        )

        return response["answer"]

    except Exception as e:
        import traceback
        print(f"Error in query: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return f"Error processing request: {e}"


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=["POST", "GET"])
def login():
    if request.method == "POST":
        email2 = request.form["email"]
        password = request.form["password"]
        user = db.collection('doctors').where('email', '==', email2).get()
        if not user:
            return redirect("/login")
        user = user[0]._data
        if user['password'] == password:
            return redirect("/ChatBot")
        else:
            return redirect("/login")
    else:
        return render_template("login.html")


@app.route('/signup', methods=["POST", "GET"])
def signup():
    if request.method == "POST":
        email = request.form["email1"]
        password = request.form["password1"]
        name = request.form['name1']
        user = db.collection('doctors').where('email', '==', email).limit(1).get()
        if len(user) != 0:
            return redirect('/signup')
        data = {'name': name, 'email': email, 'password': password}
        db.collection('doctors').add(data)
        cache['currentUser'] = email
        return redirect('/ChatBot')
    else:
        return render_template("login.html")


@app.route('/ChatBot', methods=["POST", "GET"])
def ChatBot():
    if 'chats' not in session:
        session['chats'] = []
    if request.method == "POST":
        q = str(request.form['prompt'])
        res = run_conversation(q)
        res = re.split(r'\*\*|\*', res)
        temp = [q, res]
        chats = session['chats']
        chats.append(temp)
        session['chats'] = chats
        return render_template('ChatBot.html', data=session['chats'])
    else:
        global count
        count += 1
        global session_id
        session_id = session_id[:-1]
        session_id = session_id + str(count)
        session['chats'] = []
        return render_template('ChatBot.html', data=[])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
