"""Herramientas de Investigación de Operaciones con apoyo de IA (RAG sobre un PDF).

Uso como script:   python IA.py            -> corre los 5 ejemplos
Uso como librería: from IA import resolver_lp_scipy, GrafoDijkstra, ...
                   (importar NO ejecuta los ejemplos ni carga el PDF)

La variable de entorno OPENROUTER_API_KEY se necesita para los ejemplos
con IA (4 y 5). Los solvers numéricos funcionan sin API key.
"""

import heapq
import os
import sys
from collections import defaultdict

# En Windows la consola puede usar cp1252 y fallar al imprimir emojis/acentos
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Silencia avisos inofensivos del lector de PDF (fuentes del libro)
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import pulp
import requests
from scipy.optimize import linprog

# 1. API Key: se lee de la variable de entorno, nunca se escribe en el código.
#    Soporta OPENROUTER_API_KEY (recomendado) o la variable legada "openrouter".
#    Opcional: si creás un archivo .env en esta carpeta, load_dotenv lo carga.
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("openrouter")
if not API_KEY:
    print("⚠️ No se encontró OPENROUTER_API_KEY en el entorno.")
    print("   Configurala así (PowerShell):  $env:OPENROUTER_API_KEY=\"tu-key\"")
    print("   O creá un archivo .env con:   OPENROUTER_API_KEY=tu-key")

# Ruta del PDF (en la misma carpeta que este archivo)
PDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Investigacion-Operaciones10Edicion-Frederick-S-Hillier.pdf",
)

# Límite de fragmentos para no exceder la cuota de embeddings
MAX_CHUNKS = 500


# ==================== SOLUCIONADORES NUMÉRICOS ====================

def resolver_lp_scipy(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    Resuelve problemas de programación lineal con scipy.

    Parámetros:
    - c: vector de coeficientes de la función objetivo (minimización)
    - A_ub, b_ub: restricciones de desigualdad (A_ub * x <= b_ub)
    - A_eq, b_eq: restricciones de igualdad (A_eq * x == b_eq)
    - bounds: lista de tuplas (min, max) para cada variable

    Retorna: (estado, solución, valor_objetivo, mensaje)
    """
    try:
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if result.success:
            return "ÓPTIMO", result.x, result.fun, result.message
        return "NO_ÓPTIMO", None, None, result.message
    except Exception as e:
        return "ERROR", None, None, str(e)


def resolver_lp_pulp(problema_tipo="maximizar"):
    """
    Crea un problema de LP/IP con PuLP.
    Retorna un objeto problema listo para agregar variables y restricciones.
    """
    if problema_tipo == "maximizar":
        return pulp.LpProblem("Problema_IO", pulp.LpMaximize)
    return pulp.LpProblem("Problema_IO", pulp.LpMinimize)


def agregar_variable_pulp(prob, nombre, lowBound=0, upBound=None, cat="Continuous"):
    """
    Agrega una variable al problema de PuLP.
    cat: 'Continuous', 'Integer', 'Binary'
    """
    return pulp.LpVariable(nombre, lowBound=lowBound, upBound=upBound, cat=cat)


def resolver_pulp(prob):
    """
    Resuelve el problema de PuLP.
    Retorna: (estado, dict_de_variables, valor_objetivo)
    """
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] == "Optimal":
        resultados = {v.name: v.varValue for v in prob.variables()}
        return "ÓPTIMO", resultados, pulp.value(prob.objective)
    return "NO_ÓPTIMO", None, None


# ==================== GRAFOS ====================

class GrafoDijkstra:
    """Grafo con algoritmo de Dijkstra para rutas más cortas."""

    def __init__(self):
        self.grafo = defaultdict(list)

    def agregar_arista(self, origen, destino, peso):
        """Agrega una arista dirigida."""
        self.grafo[origen].append((destino, peso))

    def agregar_arista_no_dirigida(self, nodo1, nodo2, peso):
        """Agrega una arista no dirigida."""
        self.grafo[nodo1].append((nodo2, peso))
        self.grafo[nodo2].append((nodo1, peso))

    def ruta_mas_corta(self, inicio, fin):
        """
        Encuentra la ruta más corta usando Dijkstra.
        Retorna: (distancia_minima, ruta) o (None, None) si no existe ruta.
        """
        distancias = {nodo: float("infinity") for nodo in self.grafo}
        distancias[inicio] = 0
        padres = {nodo: None for nodo in self.grafo}

        # Cola de prioridad (distancia, nodo)
        pq = [(0, inicio)]
        visitados = set()

        while pq:
            distancia_actual, nodo_actual = heapq.heappop(pq)

            if nodo_actual in visitados:
                continue
            visitados.add(nodo_actual)

            if nodo_actual == fin:
                break

            for vecino, peso in self.grafo[nodo_actual]:
                distancia = distancia_actual + peso
                if distancia < distancias[vecino]:
                    distancias[vecino] = distancia
                    padres[vecino] = nodo_actual
                    heapq.heappush(pq, (distancia, vecino))

        if distancias[fin] == float("infinity"):
            return None, None

        # Reconstruir ruta
        ruta = []
        nodo_actual = fin
        while nodo_actual is not None:
            ruta.insert(0, nodo_actual)
            nodo_actual = padres[nodo_actual]

        return distancias[fin], ruta

    def mostrar_grafo(self):
        """Muestra la estructura del grafo."""
        for nodo in self.grafo:
            print(f"{nodo} -> {dict(self.grafo[nodo])}")


# ==================== APOYO CON OPENROUTER (RAG sobre el PDF) ====================

# Endpoint y modelos de OpenRouter (usamos tu API key de OpenRouter)
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4o-mini"


class OpenRouterEmbeddings:
    """Genera embeddings llamando directo a la API de OpenRouter (compatible OpenAI)."""

    def __init__(self, api_key, model=EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model

    def embed_query(self, text):
        """Embedding de un solo texto (para la consulta del usuario)."""
        return self._embed([text])[0]

    def embed_documents(self, texts, batch_size=100):
        """Embedding de varios textos en lotes (mucho más rápido que 1 por 1)."""
        out = []
        for i in range(0, len(texts), batch_size):
            lote = texts[i:i + batch_size]
            print(f"  embeddings {i + 1}-{i + len(lote)}/{len(texts)}...", flush=True)
            out.extend(self._embed(lote))
        return out

    def _embed(self, texts):
        response = requests.post(
            f"{OPENROUTER_BASE}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]


def cargar_pdf(pdf_path=PDF_PATH):
    """Carga el PDF y lo divide en fragmentos. Retorna la lista de chunks."""
    # Imports perezosos: langchain es pesado y solo se usa en esta parte
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not os.path.exists(pdf_path):
        print(f"❌ No se encontró el archivo en {pdf_path}")
        return []

    loader = PyPDFLoader(pdf_path)
    documentos_pdf = loader.load()
    print(f"✅ PDF cargado: {len(documentos_pdf)} páginas")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = text_splitter.split_documents(documentos_pdf)
    print(f"✅ {len(chunks)} fragmentos creados")
    return chunks


def construir_base_vectorial(chunks, embeddings_model, max_chunks=MAX_CHUNKS):
    """Genera embeddings para los fragmentos. Retorna un DataFrame."""
    chunks = chunks[:max_chunks]
    if not chunks:
        print("⚠️ No hay fragmentos para embedir.")
        return pd.DataFrame()
    print(f"🔄 Generando embeddings de {len(chunks)} fragmentos (esto puede tardar unos segundos)...")
    textos = [c.page_content for c in chunks]
    vectores = embeddings_model.embed_documents(textos)
    data = [
        {
            "text": chunks[i].page_content,
            "embedding": vectores[i],
            "page": chunks[i].metadata.get("page"),
        }
        for i in range(len(chunks))
    ]
    print("✅ Base vectorial lista")
    return pd.DataFrame(data)


def buscar_contexto_relevante(query, dataframe, model, k=4):
    """Busca los k fragmentos más relevantes para la consulta (producto punto)."""
    query_vector = model.embed_query(query)
    dataframe["score"] = dataframe["embedding"].apply(lambda x: np.dot(query_vector, x))
    top_chunks = dataframe.sort_values(by="score", ascending=False).head(k)
    return "\n---\n".join(top_chunks["text"].tolist())


def preguntar_io_con_numerico(pregunta_usuario, dataframe, embeddings_model):
    """Pregunta al modelo usando como contexto los fragmentos relevantes del PDF."""
    contexto = buscar_contexto_relevante(pregunta_usuario, dataframe, embeddings_model)

    prompt = f"""
Eres un experto en Investigación de Operaciones. Analiza el siguiente problema.

CONTEXTO (teoría extraída del PDF):
{contexto}

PROBLEMA:
{pregunta_usuario}

INSTRUCCIONES:
1. Clasifica el tipo de problema (LP, Colas, Redes, PERT)
2. Presenta la formulación matemática completa
3. Explica paso a paso la resolución
4. Da los resultados numéricos finales

Si es un problema de redes, indica claramente nodos y aristas.
"""

    response = requests.post(
        f"{OPENROUTER_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "IA-Investigacion-Operaciones",
        },
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ==================== EJEMPLOS DE USO ====================

def ejemplo_scipy():
    print("\n" + "=" * 60)
    print("EJEMPLO 1: Programación Lineal con scipy.optimize.linprog")
    print("=" * 60)

    # Problema: Maximizar 40x1 + 30x2
    # Sujeto a: 2x1 + x2 <= 100 (horas máquina)
    #           x1 + 2x2 <= 80  (horas mano de obra)
    #           x1, x2 >= 0

    # Como linprog minimiza, multiplicamos por -1 para maximizar
    c = [-40, -30]
    A_ub = [[2, 1], [1, 2]]
    b_ub = [100, 80]
    bounds = [(0, None), (0, None)]

    estado, solucion, valor, mensaje = resolver_lp_scipy(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if estado == "ÓPTIMO":
        print(f"Solución óptima: x1 = {solucion[0]:.2f}, x2 = {solucion[1]:.2f}")
        print(f"Valor máximo de Z = {-valor:.2f}")  # Negativo porque minimizamos el negativo
        print(f"Estado: {mensaje}")
    else:
        print(f"No se encontró solución óptima: {mensaje}")


def ejemplo_pulp():
    print("\n" + "=" * 60)
    print("EJEMPLO 2: Programación Lineal Entera con PuLP")
    print("=" * 60)

    # Problema de producción con variables enteras
    prob = resolver_lp_pulp("maximizar")

    # Variables enteras (cantidad de productos)
    x1 = agregar_variable_pulp(prob, "Producto_A", lowBound=0, cat="Integer")
    x2 = agregar_variable_pulp(prob, "Producto_B", lowBound=0, cat="Integer")

    # Función objetivo: Maximizar 40x1 + 30x2
    prob += 40 * x1 + 30 * x2

    # Restricciones
    prob += 2 * x1 + x2 <= 100  # Horas máquina
    prob += x1 + 2 * x2 <= 80   # Horas mano de obra

    estado, resultados, valor_objetivo = resolver_pulp(prob)

    if estado == "ÓPTIMO":
        print("Solución óptima entera:")
        for var, val in resultados.items():
            print(f"  {var} = {val}")
        print(f"Valor máximo de Z = {valor_objetivo:.2f}")
    else:
        print("No se encontró solución óptima")


def ejemplo_dijkstra():
    print("\n" + "=" * 60)
    print("EJEMPLO 3: Ruta más corta con Dijkstra")
    print("=" * 60)

    grafo = GrafoDijkstra()
    grafo.agregar_arista_no_dirigida("A", "B", 4)
    grafo.agregar_arista_no_dirigida("A", "C", 2)
    grafo.agregar_arista_no_dirigida("B", "C", 1)
    grafo.agregar_arista_no_dirigida("B", "D", 5)
    grafo.agregar_arista_no_dirigida("C", "D", 8)
    grafo.agregar_arista_no_dirigida("C", "E", 10)
    grafo.agregar_arista_no_dirigida("D", "E", 2)

    print("Estructura del grafo:")
    grafo.mostrar_grafo()

    distancia, ruta = grafo.ruta_mas_corta("A", "E")

    if distancia is not None:
        print("\nRuta más corta de A a E:")
        print(f"  Ruta: {' -> '.join(ruta)}")
        print(f"  Distancia total: {distancia}")
    else:
        print("No existe ruta entre los nodos")


def modo_preguntas(dataframe, embeddings_model):
    """Modo interactivo: el usuario escribe preguntas sobre el PDF hasta escribir 'salir'."""
    print("\n" + "=" * 60)
    print("MODO PREGUNTAS: escribí tu consulta y Enter (o 'salir' para terminar)")
    print("=" * 60)
    while True:
        try:
            pregunta = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in ("salir", "exit", "q"):
            break
        if not pregunta:
            continue
        try:
            respuesta = preguntar_io_con_numerico(pregunta, dataframe, embeddings_model)
            print("\n" + respuesta + "\n")
        except Exception as e:
            print(f"❌ Ocurrió un error: {e}\n")


def ejemplos_gemini():
    """Ejemplos 4 y 5: requieren OPENROUTER_API_KEY. Retorna sin hacer nada si no hay key."""
    if not API_KEY:
        print("\n⚠️ Ejemplos 4 y 5 omitidos: configurá OPENROUTER_API_KEY para usarlos.")
        print("   Guía: https://openrouter.ai/keys")
        return

    embeddings_model = OpenRouterEmbeddings(api_key=API_KEY)
    chunks = cargar_pdf()
    if not chunks:
        return
    df = construir_base_vectorial(chunks, embeddings_model)

    print("\n" + "=" * 60)
    print("EJEMPLO 4: Problema de Redes con Gemini (Flujo máximo)")
    print("=" * 60)

    pregunta_redes = """
Encontrar el flujo máximo desde el nodo origen S hasta el nodo destino T en la siguiente red:

S -> A: capacidad 10
S -> B: capacidad 5
A -> B: capacidad 15
A -> C: capacidad 10
B -> C: capacidad 5
B -> T: capacidad 10
C -> T: capacidad 10
"""
    print(preguntar_io_con_numerico(pregunta_redes, df, embeddings_model))

    print("\n" + "=" * 60)
    print("EJEMPLO 5: PERT/CPM con datos del problema")
    print("=" * 60)

    pregunta_pert = """
Un proyecto tiene las siguientes actividades:
A: duración 3 días, predecesores: ninguna
B: duración 5 días, predecesores: A
C: duración 2 días, predecesores: A
D: duración 4 días, predecesores: B, C

Calcular:
- Tiempo early y late de cada evento
- Holguras
- Ruta crítica
"""
    print(preguntar_io_con_numerico(pregunta_pert, df, embeddings_model))

    # Modo interactivo: el usuario hace sus propias preguntas sobre el PDF
    modo_preguntas(df, embeddings_model)


def main():
    ejemplo_scipy()
    ejemplo_pulp()
    ejemplo_dijkstra()
    ejemplos_gemini()


if __name__ == "__main__":
    main()
