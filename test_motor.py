from config import get_llm
llm = get_llm()
print(llm.invoke("Decime 'hola' en una sola palabra.").content)