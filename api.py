"""API REST para consultar la base vectorial del libro de Investigación de Operaciones.

Endpoints:
- GET  /health   -> estado del servicio y cantidad de documentos indexados.
- POST /buscar   -> recupera los fragmentos más relevantes (no consume tokens de IA).
- POST /preguntar -> recupera contexto local y genera una respuesta con OpenRouter.

Uso:
    python api.py

Variables de entorno:
    OPENROUTER_API_KEY  (requerida solo para /preguntar)
    CHROMA_PATH         (opcional, default "chroma_db")
    EMBEDDING_MODEL     (opcional, default "all-MiniLM-L6-v2")
    CHAT_MODEL          (opcional, default "openai/gpt-4o-mini")
"""

import json
import os
import re
import sys

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from solver import (
    aplicar_cambios_modelo,
    es_infactible,
    formatear_cpm_matriz,
    formatear_modelo_matriz,
    resolver_cpm,
    resolver_lp_completo,
)
from vector_store import (
    buscar_contexto,
    format_context,
    get_chroma_client,
    get_collection,
    get_embedding_function,
    get_env,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"

app = FastAPI(
    title="IA Investigación de Operaciones - API RAG",
    description="Consultá el libro de IO usando una base vectorial local.",
    version="1.0.0",
)

# Cargamos la colección una sola vez al arrancar el servidor
print("🔄 Cargando base vectorial...")
client = get_chroma_client()
embedding_function = get_embedding_function()
collection = get_collection(client, embedding_function)
document_count = collection.count()
print(f"✅ Base vectorial lista: {document_count} documentos")


class BuscarRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Consulta en lenguaje natural")
    k: int = Field(4, ge=1, le=20, description="Cantidad de fragmentos a recuperar")


class PreguntarRequest(BaseModel):
    pregunta: str = Field(..., min_length=1, description="Pregunta para responder")
    k: int = Field(4, ge=1, le=20, description="Cantidad de fragmentos de contexto")
    modelo: str = Field(DEFAULT_CHAT_MODEL, description="Modelo de chat en OpenRouter")


class ResolverRequest(BaseModel):
    problema: str = Field(..., min_length=1, description="Problema a resolver")
    k: int = Field(4, ge=1, le=20, description="Cantidad de fragmentos de contexto")
    modelo: str = Field(DEFAULT_CHAT_MODEL, description="Modelo de chat en OpenRouter")


@app.get("/")
def root():
    """Página de bienvenida con los endpoints disponibles."""
    return {
        "mensaje": "API de IA - Investigación de Operaciones",
        "endpoints": {
            "docs_interactivos": "/docs",
            "health": "/health",
            "buscar": "POST /buscar",
            "preguntar": "POST /preguntar",
        },
    }


@app.get("/health")
def health():
    """Devuelve el estado del servicio y la cantidad de documentos indexados."""
    return {
        "status": "ok",
        "documentos_indexados": document_count,
        "chroma_path": get_env("CHROMA_PATH", "chroma_db"),
        "embedding_model": get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    }


@app.post("/buscar")
def buscar(request: BuscarRequest):
    """Busca los fragmentos más relevantes de forma local. No consume tokens de IA."""
    try:
        resultados = buscar_contexto(request.query, collection, k=request.k)
        return {
            "query": request.query,
            "resultados": resultados,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la búsqueda: {e}")


@app.post("/preguntar")
def preguntar(request: PreguntarRequest):
    """Recupera contexto local y genera una respuesta usando OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("openrouter")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Falta OPENROUTER_API_KEY. Configurala en el entorno o en .env",
        )

    try:
        resultados = buscar_contexto(request.pregunta, collection, k=request.k)
        contexto = format_context(resultados)

        prompt = f"""Eres un experto en Investigación de Operaciones. Analizá el siguiente problema.

CONTEXTO (teoría extraída del PDF):
{contexto}

PROBLEMA:
{request.pregunta}

INSTRUCCIONES:
1. Clasificá el tipo de problema (LP, Colas, Redes, PERT, etc.).
2. Presentá la formulación matemática completa cuando aplique.
3. Explicá paso a paso la resolución.
4. Dá los resultados numéricos finales.

Si es un problema de redes, indicá claramente nodos y aristas.
"""

        response = requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "IA-Investigacion-Operaciones",
            },
            json={
                "model": request.modelo,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=120,
        )
        response.raise_for_status()
        respuesta = response.json()["choices"][0]["message"]["content"]

        return {
            "pregunta": request.pregunta,
            "modelo": request.modelo,
            "contexto_usado": resultados,
            "respuesta": respuesta,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la respuesta: {e}")


def _extraer_json(texto: str):
    """Extrae el primer bloque JSON válido de un texto, con o sin markdown."""
    # Primero intenta encontrar JSON dentro de ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Si no, busca el primer objeto JSON plano
    match = re.search(r"(\{.*\})", texto, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError("No se encontró un JSON válido en la respuesta")


def _buscar_contexto_reparacion(collection, k: int = 4):
    """Busca en la base vectorial teoría sobre factibilidad y sensibilidad."""
    queries = [
        "infactibilidad programación lineal",
        "análisis de sensibilidad restricciones",
        "hacer factible un modelo de programación lineal",
    ]
    vistos = set()
    resultados = []
    for query in queries:
        for r in buscar_contexto(query, collection, k=k):
            texto = r.get("text", "")
            if texto and texto not in vistos:
                vistos.add(texto)
                resultados.append(r)
    return resultados


def _intentar_reparar_infactibilidad(datos: dict, solucion: dict, modelo: str, collection, api_key: str):
    """Consulta al LLM por cambios para hacer factible un LP infactible.

    Retorna un diccionario `propuesta_factibilidad` listo para incluir en la
    respuesta del endpoint /resolver.
    """
    resultados_reparacion = _buscar_contexto_reparacion(collection)
    contexto_reparacion = format_context(resultados_reparacion)

    prompt_reparacion = f"""Sos un experto en Investigación de Operaciones. Te paso un modelo de Programación Lineal que resultó infactible según el solver HiGHS.

MODELO LP ORIGINAL (datos extraídos):
{json.dumps(datos, ensure_ascii=False, indent=2)}

MENSAJE DE INFACTIBILIDAD DEL SOLVER:
{solucion.get('message', 'No disponible')}

CONTEXTO TEÓRICO DEL LIBRO SOBRE FACTIBILIDAD Y ANÁLISIS DE SENSIBILIDAD:
{contexto_reparacion}

INSTRUCCIONES ESTRICTAS:
1. Proponé cambios CONSERVADORES para hacer el problema factible. Preferí relajar el lado derecho (RHS) de restricciones o ajustar bounds antes que modificar coeficientes o quitar restricciones.
2. Si no encontrás cambios viables, devolvé {{"factible_con_cambios": false, "cambios": [], "explicacion": "..."}}.
3. Si encontrás cambios, devolvé ÚNICAMENTE un JSON válido con esta estructura exacta y sin texto adicional:

{{
  "factible_con_cambios": true,
  "cambios": [
    {{"tipo": "restriccion_ub", "indice": 0, "nuevo_b": 120, "justificacion": "..."}},
    {{"tipo": "restriccion_lb", "indice": 1, "nuevo_b": 5, "justificacion": "..."}},
    {{"tipo": "restriccion_eq", "indice": 0, "nuevo_b": 10, "justificacion": "..."}},
    {{"tipo": "bound", "variable": 1, "min": 0, "max": 50, "justificacion": "..."}}
  ],
  "explicacion": "..."
}}

Los tipos permitidos son: "bound", "restriccion_ub", "restriccion_lb", "restriccion_eq".
No incluyas texto fuera del JSON. No uses comentarios dentro del JSON.
"""

    try:
        response = requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "IA-Investigacion-Operaciones",
            },
            json={
                "model": modelo,
                "messages": [{"role": "user", "content": prompt_reparacion}],
                "temperature": 0.1,
            },
            timeout=120,
        )
        response.raise_for_status()
        raw_reparacion = response.json()["choices"][0]["message"]["content"]
        propuesta = _extraer_json(raw_reparacion)
    except Exception as e:
        return {
            "contexto_reparacion": resultados_reparacion,
            "cambios_propuestos": [],
            "datos_modificados": None,
            "solucion_reparada": None,
            "explicacion": f"No se pudo obtener una propuesta de reparación: {e}",
            "error": str(e),
        }

    cambios = propuesta.get("cambios", [])
    if not isinstance(cambios, list):
        cambios = []

    datos_modificados = aplicar_cambios_modelo(datos, cambios)
    solucion_reparada = resolver_lp_completo(datos_modificados)

    advertencia = None
    if es_infactible(solucion_reparada):
        advertencia = "La propuesta de reparación no logró hacer factible el problema. Los cambios sugeridos pueden no ser suficientes o el modelo requiere un análisis más profundo."

    propuesta_factibilidad = {
        "contexto_reparacion": resultados_reparacion,
        "cambios_propuestos": cambios,
        "datos_modificados": datos_modificados,
        "solucion_reparada": solucion_reparada,
        "matriz_reparada": formatear_modelo_matriz(datos_modificados, solucion_reparada),
        "explicacion": propuesta.get("explicacion", ""),
    }
    if advertencia:
        propuesta_factibilidad["advertencia"] = advertencia

    return propuesta_factibilidad


@app.post("/resolver")
def resolver(request: ResolverRequest):
    """Consulta la base vectorial, pide a la IA que extraiga el modelo LP y lo resuelve numéricamente."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("openrouter")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Falta OPENROUTER_API_KEY. Configurala en el entorno o en .env",
        )

    try:
        resultados = buscar_contexto(request.problema, collection, k=request.k)
        contexto = format_context(resultados)

        prompt = f"""Sos un experto en Investigación de Operaciones. Analizá el siguiente problema, clasificálo y devolvé ÚNICAMENTE un JSON válido con la estructura exacta que corresponda.

CONTEXTO TEÓRICO DEL LIBRO:
{contexto}

PROBLEMA:
{request.problema}

INSTRUCCIONES:
- Si el problema es de Programación Lineal (LP), devolvé:
  {{
    "tipo": "LP",
    "objetivo": "minimizar" o "maximizar",
    "c": [coeficientes de la función objetivo],
    "A_ub": [[fila1], [fila2], ...] para restricciones de la forma <=,
    "b_ub": [valores],
    "A_lb": [[fila1], [fila2], ...] para restricciones de la forma >= (opcional),
    "b_lb": [valores] (opcional),
    "A_eq": [[fila1], ...] para restricciones de la forma = (opcional),
    "b_eq": [valores] (opcional),
    "bounds": [[0, null], [0, null], ...] (usa null para sin límite superior),
    "explicacion": "breve explicación de qué tipo de problema es y qué método se va a usar"
  }}
  NO conviertas >= a <= manualmente. Dejá las restricciones >= en A_lb/b_lb exactamente como aparecen en el problema.
- Si el problema es de CPM/PERT (ruta crítica, redes de actividades con predecesores), devolvé:
  {{
    "tipo": "CPM",
    "actividades": [
      {{"id": "A", "nombre": "Requerimientos", "duracion": 3, "predecesores": []}},
      {{"id": "B", "nombre": "Diseño BD", "duracion": 4, "predecesores": ["A"]}},
      {{"id": "C", "nombre": "Codificación", "duracion": 6, "predecesores": ["A"]}},
      {{"id": "D", "nombre": "Pruebas", "duracion": 2, "predecesores": ["B", "C"]}}
    ],
    "explicacion": "breve explicación de qué tipo de problema es y qué método se va a usar"
  }}
- Si NO es LP ni CPM/PERT, devolvé:
  {{
    "tipo": "OTRO",
    "explicacion": "explicá por qué no es LP ni CPM/PERT y sugerí cómo resolverlo"
  }}
- NO incluyas texto fuera del JSON. NO uses comentarios dentro del JSON.
- Asegurate de que todos los coeficientes y duraciones sean números, que las matrices LP tengan dimensiones consistentes, y que los ids de actividades CPM sean únicos.
"""

        response = requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "IA-Investigacion-Operaciones",
            },
            json={
                "model": request.modelo,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]

        try:
            datos = _extraer_json(raw)
        except Exception as e:
            return {
                "problema": request.problema,
                "modelo": request.modelo,
                "error_parseo": f"No se pudo parsear la respuesta de la IA: {e}",
                "respuesta_cruda": raw,
            }

        tipo = datos.get("tipo")

        if tipo == "CPM":
            try:
                solucion = resolver_cpm(datos.get("actividades", []))
            except Exception as e:
                return {
                    "problema": request.problema,
                    "modelo": request.modelo,
                    "tipo_detectado": "CPM",
                    "error_solver": f"Error al resolver CPM: {e}",
                    "datos_extraidos": datos,
                    "contexto_usado": resultados,
                }

            return {
                "problema": request.problema,
                "modelo": request.modelo,
                "tipo_detectado": "CPM",
                "datos_extraidos": datos,
                "solucion_numerica": solucion,
                "matriz": formatear_cpm_matriz(solucion),
                "contexto_usado": resultados,
            }

        if tipo != "LP":
            return {
                "problema": request.problema,
                "modelo": request.modelo,
                "tipo_detectado": datos.get("tipo", "DESCONOCIDO"),
                "explicacion": datos.get("explicacion", "No se proporcionó explicación."),
                "contexto_usado": resultados,
            }

        solucion = resolver_lp_completo(datos)

        respuesta = {
            "problema": request.problema,
            "modelo": request.modelo,
            "tipo_detectado": "LP",
            "datos_extraidos": datos,
            "solucion_numerica": solucion,
            "matriz": formatear_modelo_matriz(datos, solucion),
            "contexto_usado": resultados,
        }

        if es_infactible(solucion):
            propuesta = _intentar_reparar_infactibilidad(
                datos=datos,
                solucion=solucion,
                modelo=request.modelo,
                collection=collection,
                api_key=api_key,
            )
            respuesta["propuesta_factibilidad"] = propuesta

        return respuesta
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al resolver el problema: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
