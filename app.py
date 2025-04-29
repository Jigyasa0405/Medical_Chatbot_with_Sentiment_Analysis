from flask import Flask, render_template, jsonify, request
from src.helper import (
    download_hugging_face_embeddings,
    fetch_and_plot_disease_data,
    record_audio,
    transcribe_audio
)
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load embeddings and retriever
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up LLaMA via Groq
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    max_tokens=500
)

# Prompt setup
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion: {input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User message:", msg)

    # Classify query as graph or text
    intent_response = llm.invoke(f"What is the intent of this query: '{msg}'? Answer with only 'graph_query' or 'text_query'.")
    intent = intent_response.content.strip()
    print("Intent:", intent)

    if intent == "graph_query":
        graph_path = fetch_and_plot_disease_data(msg)
        relative_path = os.path.basename(graph_path)  # only the filename
        return jsonify({"image_url": f"/static/{relative_path}"})

    response = rag_chain.invoke({"input": msg})
    print("RAG Response:", response["answer"])
    return str(response["answer"])

@app.route("/voice", methods=["POST"])
def voice_chat():
    audio_file = record_audio()
    user_query = transcribe_audio(audio_file)
    print("Voice input transcribed:", user_query)

    # Classify intent for voice input
    intent_response = llm.invoke(f"What is the intent of this query: '{user_query}'? Answer with only 'graph_query' or 'text_query'.")
    intent = intent_response.content.strip()
    print("Voice Intent:", intent)

    if intent == "graph_query":
        graph_path = fetch_and_plot_disease_data(user_query)
        return jsonify({
            "question": user_query,
            "answer": "Graph generated.",
            "image_url": graph_path
        })

    response = rag_chain.invoke({"input": user_query})
    return jsonify({
        "question": user_query,
        "answer": response["answer"]
    })

@app.route("/test-graph")
def test_graph():
    path = fetch_and_plot_disease_data("Show me the graph of dengue")
    return jsonify({"image_url": path})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
