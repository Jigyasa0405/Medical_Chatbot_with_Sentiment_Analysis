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
from transformers import pipeline   # ✅ for sentiment analysis
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

# Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Detect user sentiment (positive / negative / neutral)."""
    result = sentiment_pipeline(text)[0]
    label = result["label"].lower()
    if label not in ["positive", "negative"]:
        return "neutral"
    return label

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

    # Step 1: Sentiment analysis
    user_sentiment = analyze_sentiment(msg)
    print("Detected Sentiment:", user_sentiment)

    # Step 2: Classify query intent
    intent_response = llm.invoke(
        f"What is the intent of this query: '{msg}'? Answer only with 'graph_query' or 'text_query'."
    )
    intent = intent_response.content.strip()
    print("Intent:", intent)

    # Step 3: Handle graph queries
    if intent == "graph_query":
        graph_path = fetch_and_plot_disease_data(msg)
        relative_path = os.path.basename(graph_path)
        return jsonify({"image_url": f"/static/{relative_path}"})

    # Step 4: Adjust tone based on sentiment
    if user_sentiment == "negative":
        tone_prefix = "The user seems concerned or worried. Please respond calmly, reassuringly, and in an empathetic tone. "
    elif user_sentiment == "positive":
        tone_prefix = "The user seems positive. Respond in an encouraging and informative tone. "
    else:
        tone_prefix = "Provide a clear and helpful medical response. "

    # Step 5: Generate RAG response with tone context
    response = rag_chain.invoke({"input": tone_prefix + msg})
    print("RAG Response:", response["answer"])
    return str(response["answer"])

@app.route("/voice", methods=["POST"])
def voice_chat():
    audio_file = record_audio()
    user_query = transcribe_audio(audio_file)
    print("Voice input transcribed:", user_query)

    # Analyze sentiment for voice input
    user_sentiment = analyze_sentiment(user_query)
    print("Voice Sentiment:", user_sentiment)

    # Classify intent
    intent_response = llm.invoke(
        f"What is the intent of this query: '{user_query}'? Answer only with 'graph_query' or 'text_query'."
    )
    intent = intent_response.content.strip()
    print("Voice Intent:", intent)

    if intent == "graph_query":
        graph_path = fetch_and_plot_disease_data(user_query)
        return jsonify({
            "question": user_query,
            "answer": "Graph generated.",
            "image_url": graph_path
        })

    # Add tone to voice response
    if user_sentiment == "negative":
        tone_prefix = "The user sounds worried. Reply empathetically and clearly. "
    elif user_sentiment == "positive":
        tone_prefix = "The user sounds positive. Reply encouragingly. "
    else:
        tone_prefix = "Provide a clear and helpful medical response. "

    response = rag_chain.invoke({"input": tone_prefix + user_query})
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
