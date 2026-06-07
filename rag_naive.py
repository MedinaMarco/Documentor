from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import get_llm, get_embeddings

PERSIST_DIR = "chroma_db"

PROMPT = ChatPromptTemplate.from_template(
    """Sos un asistente que responde EXCLUSIVAMENTE con la información del contexto.
Si la respuesta no está en el contexto, decí: "No encontré esa información en los documentos."
No inventes. Sé claro y conciso.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
)

def format_docs(docs):
    """Une los chunks recuperados en un solo bloque de texto."""
    return "\n\n".join(d.page_content for d in docs)


def build_chain():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    # El retriever: dada una pregunta, trae los 4 chunks más parecidos.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = get_llm()

    # LCEL (LangChain Expression Language): se lee como una tubería.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    chain = build_chain()
    print("DocuMentor (RAG ingenuo). Escribí 'salir' para terminar.\n")
    while True:
        q = input("Pregunta: ")
        if q.lower() == "salir":
            break
        print("\n" + chain.invoke(q) + "\n")