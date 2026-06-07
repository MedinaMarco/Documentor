import os
from dotenv import load_dotenv

load_dotenv()

ENGINE = os.getenv("ENGINE", "ollama")


def get_llm():
    """Devuelve el modelo de lenguaje según el motor elegido."""
    if ENGINE == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.1:1b", temperature=0)
    if ENGINE == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if ENGINE == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Revisá el ID de modelo vigente en la doc de Anthropic (cambian seguido)
        return ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)
    raise ValueError(f"ENGINE desconocido: {ENGINE}")


def get_embeddings():
    """Devuelve el modelo de embeddings según el motor elegido."""
    if ENGINE == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text-v2-moe")
    if ENGINE == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    if ENGINE == "anthropic":
        # Anthropic no ofrece embeddings propios: usamos uno local gratuito
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    raise ValueError(f"ENGINE desconocido: {ENGINE}")