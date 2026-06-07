from config import get_embeddings

emb = get_embeddings()

frases = [
    "El perro corre por el parque",
    "Un cachorro juega en la plaza",
    "La ecuación describe el flujo de calor",
]
pregunta = "mascota jugando afuera"


vectores = emb.embed_documents(frases)
v_pregunta = emb.embed_query(pregunta)


def coseno(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)

print(f"Pregunta: '{pregunta}'\n")
for frase, vec in zip(frases, vectores):
    sim = coseno(v_pregunta, vec)
    print(f"{sim:.3f}  ←  {frase}")