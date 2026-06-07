from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent import build_graph
from config import get_llm

# Tu conjunto de prueba (adaptalo a TUS documentos)
GOLDEN_SET = [
    {"pregunta": "¿De qué trata el documento principal?",
     "esperado": "una descripción general del tema central"},
    {"pregunta": "¿Cuál es la conclusión más importante?",
     "esperado": "el hallazgo o recomendación clave"},
    {"pregunta": "¿que es el future learning?",
     "esperado": "Definicion completa"},
    {"pregunta": "¿que es el word embedding?",
     "esperado": "definicion completa"},
    {"pregunta": "¿cuales son y que ven las metricas de similitud?",
     "esperado": "Definicion y descripcion del tema"},
    {"pregunta": "¿que es el testrank?",
     "esperado": "Definicion"},
    {"pregunta": "¿que es el PLN?",
     "esperado": "definicion"},
    {"pregunta": "¿que se evaluo en la etapa de pre-procesamiento?",
     "esperado": "una descripción general del tema central"},
    {"pregunta": "¿que se vio en los resultados experimentales?",
     "esperado": "el hallazgo y descripcion"},
]

judge = get_llm()
judge_prompt = ChatPromptTemplate.from_template(
    """Sos un evaluador estricto. Dada una PREGUNTA, una RESPUESTA del sistema
y lo que se ESPERABA, calificá del 1 al 5 qué tan buena es la respuesta
(5 = excelente, fiel y completa; 1 = incorrecta o inventada).
Devolvé SOLO el número.

PREGUNTA: {pregunta}
ESPERADO: {esperado}
RESPUESTA: {respuesta}"""
)
judge_chain = judge_prompt | judge | StrOutputParser()


def main():
    agent = build_graph()
    puntajes = []
    for caso in GOLDEN_SET:
        result = agent.invoke({
            "question": caso["pregunta"],
            "original_question": caso["pregunta"],
            "documents": [], "generation": "", "retries": 0,
        })
        nota = judge_chain.invoke({
            "pregunta": caso["pregunta"],
            "esperado": caso["esperado"],
            "respuesta": result["generation"],
        })
        try:
            puntajes.append(int(nota.strip()[0]))
        except (ValueError, IndexError):
            puntajes.append(0)
        print(f"[{nota.strip()[:1]}/5] {caso['pregunta']}")

    if puntajes:
        print(f"\n📊 Puntaje promedio: {sum(puntajes)/len(puntajes):.2f}/5")


if __name__ == "__main__":
    main()