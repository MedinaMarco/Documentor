from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from config import get_llm, get_embeddings


PERSIST_DIR = "chroma_db"
MAX_RETRIES = 2  # cuántas veces reformula antes de rendirse

class GraphState(TypedDict):
    question: str           # la pregunta ACTUAL (puede reescribirse)
    original_question: str  # la pregunta ORIGINAL del usuario
    documents: List[Document]
    generation: str
    retries: int
    
    
embeddings = get_embeddings()
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = get_llm()

def retrieve(state: GraphState):
    docs = retriever.invoke(state["question"])
    return {"documents": docs}

class GradeDocument(BaseModel):
    relevante: str = Field(description="Respondé 'si' o 'no'")

grader_prompt = ChatPromptTemplate.from_template(
    """¿El siguiente documento es relevante para responder la pregunta?
Respondé solo con 'si' o 'no'.

Documento:
{document}

Pregunta: {question}"""
)
doc_grader = grader_prompt | llm.with_structured_output(GradeDocument)

def grade_documents(state: GraphState):
    relevantes = []
    for d in state["documents"]:
        score = doc_grader.invoke(
            {"document": d.page_content, "question": state["question"]}
        )
        if score.relevante.strip().lower() == "si":
            relevantes.append(d)
    print(f"  📋 {len(relevantes)}/{len(state['documents'])} documentos relevantes")
    return {"documents": relevantes}




rewrite_prompt = ChatPromptTemplate.from_template(
    """La búsqueda con esta pregunta no trajo buenos resultados.
Reformulala para que sea más efectiva en una búsqueda semántica.
Devolvé SOLO la nueva pregunta, sin explicaciones.

Pregunta original: {question}"""
)
rewriter = rewrite_prompt | llm | StrOutputParser()

def transform_query(state: GraphState):
    nueva = rewriter.invoke({"question": state["question"]})
    print(f"  🔄 Reformulada: {nueva}")
    return {"question": nueva, "retries": state["retries"] + 1}


gen_prompt = ChatPromptTemplate.from_template(
    """Respondé la pregunta usando SOLO el contexto.
Si el contexto no alcanza, decí honestamente que no encontraste la información.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
)
generator = gen_prompt | llm | StrOutputParser()

def generate(state: GraphState):
    context = "\n\n".join(d.page_content for d in state["documents"])
    answer = generator.invoke(
        {"context": context, "question": state["original_question"]}
    )
    return {"generation": answer}


def decide(state: GraphState):
    if state["documents"]:        # quedaron docs relevantes
        return "generate"
    if state["retries"] < MAX_RETRIES:  # sin docs, pero hay reintentos
        return "transform_query"
    return "generate"             # sin docs y sin reintentos: respondemos con elegancia


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("transform_query", transform_query)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        decide,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    g.add_edge("transform_query", "retrieve")  # el bucle de reintento
    g.add_edge("generate", END)

    return g.compile()


class Groundedness(BaseModel):
    fundamentada: str = Field(description="'si' si la respuesta se apoya en el contexto, 'no' si no")

ground_prompt = ChatPromptTemplate.from_template(
    """¿La RESPUESTA está respaldada por el CONTEXTO? Respondé 'si' o 'no'.

Contexto: {context}
Respuesta: {generation}"""
)
ground_checker = ground_prompt | llm.with_structured_output(Groundedness)


if __name__ == "__main__":
    app = build_graph()
    print("DocuMentor AGÉNTICO. Escribí 'salir' para terminar.\n")
    while True:
        q = input("Pregunta: ")
        if q.lower() == "salir":
            break
        result = app.invoke({
            "question": q,
            "original_question": q,
            "documents": [],
            "generation": "",
            "retries": 0,
        })
        print("\n" + result["generation"] + "\n")
        

