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
    """Use OpenAI to generate a final answer from question + multiple context chunks"""
    
    # Filter and combine only valid text chunks
    valid_chunks = [chunk['metadata'].get('text', '') for chunk in context_chunks if 'metadata' in chunk and 'text' in chunk['metadata']]
    context = "\n\n".join(valid_chunks)

    if not context.strip():
        return "I couldn't find enough relevant information to answer the question."

    prompt = f"""
You are a helpful assistant. Use the context below to answer the question. If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

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


def rerank_chunks(question, chunks, top_n=3):
    """
    Use OpenAI to select the most relevant chunks from Pinecone results.
    Returns the top_n most relevant chunks as decided by GPT.
    """
    context_texts = [
        chunk['metadata'].get('text', '') for chunk in chunks if 'metadata' in chunk and 'text' in chunk['metadata']
    ]

    prompt = f"""
Given the question: "{question}"

Select the top {top_n} most relevant context chunks from the list below. 
Only return the chunk numbers in descending order of relevance. 

"""

    for i, chunk_text in enumerate(context_texts):
        prompt += f"Chunk {i+1}:\n{chunk_text}\n\n"

    prompt += "Return the chunk numbers of the most relevant ones (e.g., 2, 5, 1):"

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    selected_indices = response.choices[0].message.content.strip()
    # Parse the result like: "2, 1, 3" => [1, 0, 2]
    try:
        selected_indices = [int(i.strip()) - 1 for i in selected_indices.split(",")]
        return [chunks[i] for i in selected_indices if 0 <= i < len(chunks)]
    except Exception as e:
        print("❌ Failed to parse reranked output:", e)
        return chunks[:top_n]  # fallback



def rag_chatbot_pipeline(user_question, history=None, professor=None):
    memory_summary = ""
    if history and len(history) >= 3 and len(history) % 3 == 0:
        memory_summary = summarize_history(history[:-1])
        user_question = f"Context: {memory_summary}\n\nNow answer this:\n{user_question}"

    rephrased_question = generate_chain_of_thought(user_question)
    chunks = search_pinecone(rephrased_question, professor=professor, top_k=6)
    chunks = rerank_chunks(rephrased_question, chunks, top_n=3)
    final_answer = generate_answer(user_question, chunks)

    return final_answer, memory_summary



# === Run the flow ===
if __name__ == "__main__":
    query = input("Ask a question about a professor: ")
    answer = rag_chatbot_pipeline(query)
    print("\n💬 Answer:\n", answer)
