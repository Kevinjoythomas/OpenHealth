import torch
from langchain_ollama import OllamaLLM
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
CHROMA_PATH = "./chroma"
PROMPT_TEMPLATE = """
You are a medical professional. Answer the question like a doctor would. It should consist of paragraph and conversational aspect rather than just a summary. Answer the asked question briefly. Answer in a professional tone:

{context}

---

Answer the question based on your knowledge and using the above context for help: {question}
"""

print(torch.cuda.device_count())          
print(torch.cuda.get_device_name(0))      
     
user_prompt = "hello my head is hurting and i have vomiting. what should i do"

gatePrompt = f"<|start_header_id|>system<|end_header_id|>I will now give you a question. This question should only be related to medical queries or advice. If it is related to medical queries or advice, then reply with 'True' and nothing else, no explanation, nothing, just 'True'. If it's not related to medical info, then just say 'False' and nothing else, no explanation, nothing, just 'False'. Just reply with either True or False and nothing else.<|eot_id|><|start_header_id|>user<|end_header_id|> This is the question: {user_prompt}<|eot_id|>"
gatedModel = OllamaLLM(model="llama3")
gateResult = gatedModel.invoke(gatePrompt)

if(str.lower(gateResult) == "false"):
    print("This query is not related to medical field. Please ask related queries.")
    exit(0)

embedding_function = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory=CHROMA_PATH,embedding_function=embedding_function)

results = db.similarity_search_with_score(user_prompt,k=5)

context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text,question=user_prompt)
print(prompt, "Number of db docs", len(db.get()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OllamaLLM(model="llama3.2")
response = model.invoke(prompt)
print("The response is \n",response)