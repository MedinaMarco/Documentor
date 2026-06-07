from fastapi import FastAPI
from pydantic import BaseModel
from agent import build_graph

app = FastAPI(title="DocuMentor API")
agent = build_graph()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    result = agent.invoke({
        "question": q.question, "original_question": q.question,
        "documents": [], "generation": "", "retries": 0,
    })
    return {
        "answer": result["generation"],
        "sources": [d.metadata.get("source") for d in result["documents"]],
    }