# Base de datos vectorial persistente + API (RAG local)

## Objetivo
Persistir la base vectorial del libro de Investigación de Operaciones en una carpeta local y exponer una API REST para consultarla. La búsqueda de contexto debe ser 100 % local para no consumir tokens de la API de IA; la generación de la respuesta puede usar OpenRouter opcionalmente.

## Problema
Hoy `IA.py` genera los embeddings cada vez que corre (`construir_base_vectorial` crea un `DataFrame` en memoria). Eso:
- Gasta tokens de embedding en cada ejecución.
- Es lento porque re-procesa el PDF.
- No permite consultar desde otro proceso/una API.

## Solución propuesta
- Usar **ChromaDB** con persistencia en disco (`chroma_db/`).
- Usar **embeddings locales** con `sentence-transformers` (`all-MiniLM-L6-v2`) para no depender de la API de embeddings.
- Crear un script `indexar.py` que procese el PDF una sola vez y guarde los vectores.
- Crear una API con **FastAPI** (`api.py`) con endpoints:
  - `POST /buscar` → devuelve los fragmentos más relevantes (0 tokens de IA).
  - `POST /preguntar` → recupera contexto local y genera respuesta con OpenRouter.

## Alcance
- Mantener `IA.py` existente sin romperlo (no modificar su comportamiento actual).
- Agregar tres archivos nuevos: `vector_store.py`, `indexar.py`, `api.py`.
- Actualizar `requirements.txt` con las nuevas dependencias.
- No tocar `.env` (contiene secretos); documentar las variables necesarias.

## Tareas
1. [x] Instalar dependencias: `chromadb`, `sentence-transformers`, `fastapi`, `uvicorn[standard]`, `python-multipart`.
2. [x] Crear `vector_store.py` con funciones de carga de PDF, indexación y búsqueda.
3. [x] Crear `indexar.py` para generar/actualizar la base vectorial desde el PDF.
4. [x] Crear `api.py` con endpoints `/health`, `/buscar` y `/preguntar`.
5. [x] Actualizar/crear `requirements.txt`.
6. [x] Verificar que `indexar.py` corra y persista `chroma_db/`.
7. [x] Verificar que `api.py` levante y responda correctamente.
8. [x] Agregar ruta raíz `/` para evitar confusión con 404 en navegador.
9. [x] Crear `gui.py` con interfaz gráfica para consultar la API.
10. [x] Crear `solver.py` para resolver LP numéricamente con scipy/PuLP.
11. [x] Crear endpoint `/resolver` que use IA para parsear el problema y scipy/PuLP para resolverlo.
12. [x] Actualizar `gui.py` con botón "Resolver numéricamente".
13. [x] Probar `/resolver` con el problema de gas natural del usuario (detectó infactibilidad por datos inconsistentes) y con un LP de ejemplo (resolvió correctamente).

## Criterios de aceptación
- `python indexar.py` procesa el PDF y deja la carpeta `chroma_db/` con los vectores.
- `python api.py` levanta el servidor en `http://localhost:8000`.
- `POST /buscar` devuelve JSON con los `k` fragmentos más relevantes.
- `POST /preguntar` devuelve una respuesta generada por OpenRouter usando contexto local.
- No se regeneran embeddings en cada consulta.

## Variables de entorno
- `OPENROUTER_API_KEY` (requerida solo para `/preguntar`).
- `CHROMA_PATH` (opcional, default `chroma_db`).
- `EMBEDDING_MODEL` (opcional, default `all-MiniLM-L6-v2`).
- `CHAT_MODEL` (opcional, default `openai/gpt-4o-mini`).

## Checks
- [ ] `python indexar.py` termina sin errores.
- [ ] `python -c "import api; print('ok')"` importa la API sin errores.
- [ ] Petición de ejemplo a `/buscar` devuelve resultados.

## Notas
- El repositorio no es un repo Git; no se realizarán commits.
- Los embeddings locales descargan el modelo una vez (~80 MB) y luego funcionan offline.
