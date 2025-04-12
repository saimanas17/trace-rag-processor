import openai
from pinecone import Pinecone
from embedding.embedder import get_openai_embedding
from dotenv import load_dotenv
import os

load_dotenv()


# === Configuration ===
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")


def generate_chain_of_thought(question):
    """Generate a reasoning-based reformulation of the user question"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that reformulates user questions into well-reasoned queries for research or academic purposes."},
            {"role": "user", "content": f"Break this question into a clear reasoning chain for better search: '{question}'"}
        ]
    )
    return response.choices[0].message.content.strip()


def search_pinecone(query, professor=None, top_k=5):
    query_vector = get_openai_embedding(query)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    filter_dict = {"professor": professor} if professor else None

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )
    return results.matches



def generate_answer(question, context_chunks):
    """Use OpenAI to generate a final answer from question + context"""
    context = "\n\n".join(chunk['metadata']['text'] for chunk in context_chunks)

    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def summarize_history(chat_history):
    """Summarize earlier Q&A to preserve memory context."""
    history_text = "\n\n".join(
        f"User: {item['user']}\nAssistant: {item['bot']}" for item in chat_history
    )

    prompt = f"""Summarize the following conversation between a student and a chatbot into a few key points. Keep it factual and concise.

Conversation:
{history_text}

Summary:"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


def rag_chatbot_pipeline(user_question, history=None, professor=None):
    memory_summary = ""
    if history and len(history) >= 3 and len(history) % 3 == 0:
        memory_summary = summarize_history(history[:-1])
        user_question = f"Context: {memory_summary}\n\nNow answer this:\n{user_question}"

    rephrased_question = generate_chain_of_thought(user_question)
    chunks = search_pinecone(rephrased_question, professor=professor)
    final_answer = generate_answer(user_question, chunks)

    return final_answer, memory_summary



# === Run the flow ===
if __name__ == "__main__":
    query = input("Ask a question about a professor: ")
    answer = rag_chatbot_pipeline(query)
    print("\n💬 Answer:\n", answer)
