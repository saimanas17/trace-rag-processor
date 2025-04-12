from fastapi import FastAPI
from pydantic import BaseModel
from rag.rag_chain_of_thought import rag_chatbot_pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend access (e.g., from Streamlit, React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class QuestionRequest(BaseModel):
    question: str
    history: list[dict] = []
    professor: str = "Unknown"

# Response schema
class RAGResponse(BaseModel):
    answer: str
    summary: str | None = None

# Main endpoint
@app.post("/rag/ask", response_model=RAGResponse)
async def ask_question(req: QuestionRequest):
    answer, summary = rag_chatbot_pipeline(
        req.question,
        history=req.history,
        professor=req.professor
    )
    return {"answer": answer, "summary": summary}


@app.get("/professors")
def list_professors():
    import psycopg2
    from config.settings import DB_CONFIG

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT instructor_name FROM trace.course_info")
    names = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return sorted(names)
