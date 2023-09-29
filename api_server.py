from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

custom_prompt_template = """
The following is a friendly conversation between a human and you. 
You provides lots of specific details from its context.
Your name is Defai and you are a chatbot that is created by AI Focal. You want to help the human that you are chatting.
If you do not know the answer to a question, you truthfully says you do not know.

Current conversation:
{history}
Human: {input}
AI:
"""

prompt = PromptTemplate(input_variables=["history", "input"], template=custom_prompt_template)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
conversation = ConversationChain(
    llm=llm,
    verbose=False,
    prompt = prompt,
)

@app.route("/api/query_defai", methods=["POST"])
def query_defai():
    question = request.json['question']
    if question:
        try:
            res = conversation.predict(input=question)
            return jsonify({"response" : res})
        except Exception:
            return jsonify({"error": "An error occurred. Please try again!"}), 500
    return jsonify({"error": "Invalid request. Please provide 'question' in JSON format."}), 400

@app.route("/api/bot_initialize", methods=["POST"])
def initialize_chat_bot():
    return "1"

@app.route("/api/query_bot", methods=["POST"])
def query_chat_bot():
    return "1"

if __name__ == '__main__': 
   app.run(host="0.0.0.0", port=5000, debug=False)
