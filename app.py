import streamlit as st
from agent import build_graph

st.set_page_config(page_title="DocuMentor", page_icon="📚")
st.title("📚 DocuMentor")
st.caption("Asistente RAG agéntico · conversá con tus documentos")


@st.cache_resource
def load_agent():
    """Carga el grafo una sola vez (no en cada interacción)."""
    return build_graph()


agent = load_agent()

# Historial de la conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input del usuario
if prompt := st.chat_input("Preguntá sobre tus documentos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            result = agent.invoke({
                "question": prompt,
                "original_question": prompt,
                "documents": [],
                "generation": "",
                "retries": 0,
            })
            answer = result["generation"]
            st.markdown(answer)

            # Citado de fuentes: el detalle que da confianza
            if result["documents"]:
                with st.expander("📄 Fuentes consultadas"):
                    for i, d in enumerate(result["documents"], 1):
                        fuente = d.metadata.get("source", "documento")
                        pagina = d.metadata.get("page", "?")
                        st.caption(f"**{i}.** {fuente} — pág. {pagina}")

    st.session_state.messages.append({"role": "assistant", "content": answer})