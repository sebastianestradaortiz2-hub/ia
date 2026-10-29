"""Interfaz gráfica para consultar la API de IA de Investigación de Operaciones.

Uso:
    python gui.py

La ventana verifica que la API esté corriendo en http://localhost:8000.
Si no está corriendo, te pregunta si querés iniciarla automáticamente.
"""

import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_URL = "http://localhost:8000"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("IA - Investigación de Operaciones")
        self.root.geometry("800x700")
        self.root.minsize(600, 500)

        # Estado superior
        self.status_frame = tk.Frame(root)
        self.status_frame.pack(fill="x", padx=10, pady=5)

        self.status_label = tk.Label(self.status_frame, text="Verificando API...", fg="blue")
        self.status_label.pack(side="left")

        self.docs_label = tk.Label(self.status_frame, text="")
        self.docs_label.pack(side="right")

        # Separador
        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=5)

        # Entrada de pregunta
        input_frame = tk.LabelFrame(root, text="Tu consulta", padx=10, pady=5)
        input_frame.pack(fill="both", expand=False, padx=10, pady=5)

        self.question_text = scrolledtext.ScrolledText(input_frame, height=6, wrap=tk.WORD)
        self.question_text.pack(fill="both", expand=True)
        self.question_text.bind("<Control-Return>", lambda e: self.ask_question())

        # Botones
        button_frame = tk.Frame(root)
        button_frame.pack(fill="x", padx=10, pady=5)

        self.ask_button = tk.Button(
            button_frame,
            text="Preguntar con IA",
            command=self.ask_question,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        self.ask_button.pack(side="left", padx=5)

        self.resolve_button = tk.Button(
            button_frame,
            text="Resolver numéricamente",
            command=self.resolve_problem,
            bg="#FF9800",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        self.resolve_button.pack(side="left", padx=5)

        self.search_button = tk.Button(
            button_frame,
            text="Buscar en el libro (sin tokens)",
            command=self.search_book,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        self.search_button.pack(side="left", padx=5)

        self.clear_button = tk.Button(button_frame, text="Limpiar", command=self.clear_all)
        self.clear_button.pack(side="right", padx=5)

        # Spinner
        self.spinner = tk.Label(root, text="", fg="blue")
        self.spinner.pack(pady=2)

        # Respuesta
        output_frame = tk.LabelFrame(root, text="Respuesta", padx=10, pady=5)
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.answer_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state="disabled")
        self.answer_text.pack(fill="both", expand=True)

        # Atajo
        tip = tk.Label(root, text="Tip: Ctrl+Enter envía la pregunta", fg="gray", font=("Arial", 8))
        tip.pack(pady=2)

        self.set_buttons_state("disabled")
        self.check_api()

    def set_buttons_state(self, state):
        self.ask_button.config(state=state)
        self.resolve_button.config(state=state)
        self.search_button.config(state=state)

    def set_answer(self, text):
        self.answer_text.config(state="normal")
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, text)
        self.answer_text.config(state="disabled")

    def append_answer(self, text):
        self.answer_text.config(state="normal")
        self.answer_text.insert(tk.END, text)
        self.answer_text.config(state="disabled")

    def _formatear_matriz(self, matriz: dict) -> str:
        """Convierte el diccionario matricial del solver en una tabla tipo Excel."""
        if not isinstance(matriz, dict):
            return ""

        lineas = []

        # Función objetivo
        fo = matriz.get("funcion_objetivo", {})
        if fo:
            lineas.append("📐 FUNCIÓN OBJETIVO")
            lineas.append(self._tabla_a_texto([fo]))
            lineas.append("")

        # Restricciones
        restricciones = matriz.get("restricciones", [])
        if restricciones:
            lineas.append("📊 RESTRICCIONES (matriz A | tipo | RHS)")
            lineas.append(self._tabla_a_texto(restricciones))
            lineas.append("")

        # Bounds
        bounds = matriz.get("bounds", [])
        if bounds:
            lineas.append("🔒 BOUNDS DE VARIABLES")
            lineas.append(self._tabla_a_texto(bounds))
            lineas.append("")

        # Solución
        sol = matriz.get("solucion", {})
        if sol:
            lineas.append("✅ SOLUCIÓN")
            variables = sol.get("variables", [])
            if variables:
                lineas.append(self._tabla_a_texto(variables))
            z = sol.get("valor_objetivo")
            if z is not None:
                lineas.append(f"Valor objetivo Z = {z}")
            status = sol.get("status")
            if status:
                lineas.append(f"Estado: {status}")
            lineas.append("")

        return "\n".join(lineas)

    def _tabla_a_texto(self, filas: list) -> str:
        """Alinea columnas de una lista de dicts como una tabla tipo Excel."""
        if not filas:
            return ""

        columnas = list(filas[0].keys())
        anchos = {col: len(str(col)) for col in columnas}
        for fila in filas:
            for col in columnas:
                anchos[col] = max(anchos[col], len(str(fila.get(col, ""))))

        partes = []
        separador = "+".join("-" * (anchos[col] + 2) for col in columnas)
        separador = f"+{separador}+"

        header = " | ".join(str(col).rjust(anchos[col]) for col in columnas)
        partes.append(f"| {header} |")
        partes.append(separador)

        for fila in filas:
            linea = " | ".join(str(fila.get(col, "")).rjust(anchos[col]) for col in columnas)
            partes.append(f"| {linea} |")

        return "\n".join(partes)

    def check_api(self):
        """Verifica si la API está corriendo. Si no, pregunta si iniciarla."""
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                self.status_label.config(
                    text="✅ API conectada", fg="green"
                )
                self.docs_label.config(text=f"Documentos indexados: {data.get('documentos_indexados', '?')}")
                self.set_buttons_state("normal")
                return
        except Exception:
            pass

        self.status_label.config(text="❌ API no encontrada", fg="red")
        self.docs_label.config(text="")

        if messagebox.askyesno(
            "API no está corriendo",
            "La API no responde en http://localhost:8000.\n\n¿Querés iniciarla ahora?\n"
            "(Se va a abrir otra ventana de consola con el servidor)",
        ):
            self.start_api()
        else:
            self.set_answer(
                "La API no está corriendo.\n\n"
                "Iniciala manualmente con:\n"
                "  .venv\\Scripts\\python api.py\n\n"
                "Y después volvé a ejecutar este programa."
            )

    def start_api(self):
        """Inicia api.py en una consola nueva y espera a que esté lista."""
        self.status_label.config(text="⏳ Iniciando API...", fg="blue")
        self.set_buttons_state("disabled")

        try:
            subprocess.Popen(
                [sys.executable, os.path.join(BASE_DIR, "api.py")],
                cwd=BASE_DIR,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar la API:\n{e}")
            self.status_label.config(text="❌ Error al iniciar API", fg="red")
            return

        self.spinner.config(text="Esperando que la API esté lista...")
        threading.Thread(target=self._wait_for_api, daemon=True).start()

    def _wait_for_api(self):
        """Espera hasta que /health responda."""
        for _ in range(60):
            try:
                response = requests.get(f"{API_URL}/health", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    self.root.after(
                        0,
                        lambda: (
                            self.status_label.config(text="✅ API conectada", fg="green"),
                            self.docs_label.config(
                                text=f"Documentos indexados: {data.get('documentos_indexados', '?')}"
                            ),
                            self.set_buttons_state("normal"),
                            self.spinner.config(text=""),
                            self.set_answer("API iniciada correctamente. Ya podés hacer tu consulta."),
                        ),
                    )
                    return
            except Exception:
                pass
            time.sleep(1)

        self.root.after(
            0,
            lambda: (
                self.status_label.config(text="❌ API no inició", fg="red"),
                self.spinner.config(text=""),
                self.set_buttons_state("disabled"),
                messagebox.showerror(
                    "Error",
                    "La API no respondió después de 60 segundos.\n"
                    "Revisá la ventana del servidor para ver el error.",
                ),
            ),
        )

    def ask_question(self):
        pregunta = self.question_text.get("1.0", tk.END).strip()
        if not pregunta:
            messagebox.showwarning("Consulta vacía", "Escribí una pregunta antes de enviar.")
            return

        self.set_buttons_state("disabled")
        self.spinner.config(text="🤖 Pensando...")
        self.set_answer("")

        threading.Thread(target=self._call_preguntar, args=(pregunta,), daemon=True).start()

    def _call_preguntar(self, pregunta):
        try:
            response = requests.post(
                f"{API_URL}/preguntar",
                json={"pregunta": pregunta, "k": 4},
                timeout=120,
            )
            if response.status_code == 401:
                text = (
                    "⚠️ Falta la API key de OpenRouter.\n\n"
                    "Agregá esta línea a tu archivo .env:\n"
                    "OPENROUTER_API_KEY=tu-key\n\n"
                    "Y reiniciá la API."
                )
            elif response.status_code != 200:
                text = f"Error {response.status_code}:\n{response.text}"
            else:
                data = response.json()
                text = data.get("respuesta", "No se recibió respuesta.")
        except Exception as e:
            text = f"Error al conectar con la API:\n{e}"

        self.root.after(
            0,
            lambda: (
                self.set_answer(text),
                self.set_buttons_state("normal"),
                self.spinner.config(text=""),
            ),
        )

    def resolve_problem(self):
        problema = self.question_text.get("1.0", tk.END).strip()
        if not problema:
            messagebox.showwarning("Consulta vacía", "Escribí el problema para resolver.")
            return

        self.set_buttons_state("disabled")
        self.spinner.config(text="🔢 Resolviendo numéricamente...")
        self.set_answer("")

        threading.Thread(target=self._call_resolver, args=(problema,), daemon=True).start()

    def _call_resolver(self, problema):
        try:
            response = requests.post(
                f"{API_URL}/resolver",
                json={"problema": problema, "k": 4},
                timeout=180,
            )
            if response.status_code == 401:
                text = (
                    "⚠️ Falta la API key de OpenRouter.\n\n"
                    "Agregá esta línea a tu archivo .env:\n"
                    "OPENROUTER_API_KEY=tu-key\n\n"
                    "Y reiniciá la API."
                )
            elif response.status_code != 200:
                text = f"Error {response.status_code}:\n{response.text}"
            else:
                data = response.json()
                if "error_parseo" in data:
                    text = (
                        f"⚠️ No se pudo entender el problema.\n\n"
                        f"Error: {data['error_parseo']}\n\n"
                        f"Respuesta cruda de la IA:\n{data.get('respuesta_cruda', 'N/A')}"
                    )
                elif data.get("tipo_detectado") == "CPM":
                    datos = data.get("datos_extraidos", {})
                    sol = data.get("solucion_numerica", {})
                    matriz = data.get("matriz", {})
                    partes = []
                    partes.append(f"Tipo detectado: CPM/PERT")
                    partes.append(f"\nExplicación de la IA:\n{datos.get('explicacion', 'N/A')}")
                    partes.append(f"\n⏱️ Duración total del proyecto: {sol.get('duracion_total')}")
                    partes.append(f"🎯 Ruta crítica: {' -> '.join(sol.get('ruta_critica', []))}")

                    tabla = matriz.get("tabla_actividades", [])
                    if tabla:
                        partes.append("\n" + "=" * 60)
                        partes.append("📋 TABLA DE ACTIVIDADES CPM")
                        partes.append("=" * 60)
                        partes.append(self._tabla_a_texto(tabla))

                    caminos = sol.get("caminos", [])
                    if caminos:
                        partes.append("\n" + "=" * 60)
                        partes.append("🛤️ CAMINOS POSIBLES")
                        partes.append("=" * 60)
                        for camino_info in caminos:
                            camino = camino_info.get("camino", [])
                            duracion = camino_info.get("duracion")
                            partes.append(
                                f"{' -> '.join(camino)}  (duración: {duracion})"
                            )

                    text = "\n".join(partes)
                elif data.get("tipo_detectado") != "LP":
                    text = (
                        f"Tipo detectado: {data.get('tipo_detectado')}\n\n"
                        f"Explicación:\n{data.get('explicacion', 'N/A')}"
                    )
                else:
                    datos = data.get("datos_extraidos", {})
                    sol = data.get("solucion_numerica", {})
                    partes = []
                    partes.append(f"Tipo detectado: {datos.get('tipo')}")
                    partes.append(f"Objetivo: {datos.get('objetivo')}")
                    partes.append(f"Función objetivo (c): {datos.get('c')}")
                    partes.append(f"\nExplicación de la IA:\n{datos.get('explicacion', 'N/A')}")

                    matriz = data.get("matriz")
                    if matriz:
                        partes.append("\n" + "=" * 60)
                        partes.append("📋 MODELO Y SOLUCIÓN EN FORMATO MATRICIAL")
                        partes.append("=" * 60)
                        partes.append(self._formatear_matriz(matriz))

                    if sol.get("status") == "ÓPTIMO":
                        partes.append(f"\n✅ SOLUCIÓN ÓPTIMA")
                        partes.append(f"Variables (x1, x2, ...): {sol.get('solucion')}")
                        partes.append(f"Valor objetivo Z: {sol.get('valor_objetivo')}")
                    else:
                        partes.append(f"\n⚠️ El solver no encontró solución óptima.")
                        partes.append(f"Estado: {sol.get('status')}")
                        partes.append(f"Mensaje: {sol.get('message')}")

                    partes.append(f"\nMensaje del solver: {sol.get('message')}")

                    propuesta = data.get("propuesta_factibilidad")
                    if propuesta:
                        partes.append("\n" + "=" * 60)
                        partes.append("🔧 PROPUESTA DE REPARACIÓN DE INFACTIBILIDAD")
                        partes.append("=" * 60)
                        partes.append(f"\nExplicación de la reparación:\n{propuesta.get('explicacion', 'No disponible')}")

                        cambios = propuesta.get("cambios_propuestos", [])
                        if cambios:
                            partes.append("\nCambios propuestos:")
                            for i, cambio in enumerate(cambios, 1):
                                tipo = cambio.get("tipo", "desconocido")
                                justif = cambio.get("justificacion", "Sin justificación")
                                if tipo == "bound":
                                    var = cambio.get("variable")
                                    min_v = cambio.get("min")
                                    max_v = cambio.get("max")
                                    partes.append(f"  {i}. Bound de x{var + 1}: min={min_v}, max={max_v} — {justif}")
                                elif tipo.startswith("restriccion_"):
                                    idx = cambio.get("indice")
                                    nuevo_b = cambio.get("nuevo_b")
                                    partes.append(f"  {i}. {tipo} índice {idx}: nuevo RHS={nuevo_b} — {justif}")
                                else:
                                    partes.append(f"  {i}. {cambio} — {justif}")

                        sol_rep = propuesta.get("solucion_reparada", {})
                        matriz_rep = propuesta.get("matriz_reparada")
                        if matriz_rep:
                            partes.append("\n📋 MODELO REPARADO EN FORMATO MATRICIAL")
                            partes.append(self._formatear_matriz(matriz_rep))

                        if sol_rep:
                            partes.append("\n📊 Solución reparada:")
                            partes.append(f"Estado: {sol_rep.get('status')}")
                            if sol_rep.get("status") == "ÓPTIMO":
                                partes.append(f"Variables (x1, x2, ...): {sol_rep.get('solucion')}")
                                partes.append(f"Valor objetivo Z: {sol_rep.get('valor_objetivo')}")
                            partes.append(f"Mensaje: {sol_rep.get('message')}")

                        if "advertencia" in propuesta:
                            partes.append(f"\n⚠️ ADVERTENCIA: {propuesta['advertencia']}")

                    text = "\n".join(partes)
        except Exception as e:
            text = f"Error al conectar con la API:\n{e}"

        self.root.after(
            0,
            lambda: (
                self.set_answer(text),
                self.set_buttons_state("normal"),
                self.spinner.config(text=""),
            ),
        )

    def search_book(self):
        query = self.question_text.get("1.0", tk.END).strip()
        if not query:
            messagebox.showwarning("Consulta vacía", "Escribí algo para buscar en el libro.")
            return

        self.set_buttons_state("disabled")
        self.spinner.config(text="🔍 Buscando en el libro...")
        self.set_answer("")

        threading.Thread(target=self._call_buscar, args=(query,), daemon=True).start()

    def _call_buscar(self, query):
        try:
            response = requests.post(
                f"{API_URL}/buscar",
                json={"query": query, "k": 4},
                timeout=30,
            )
            if response.status_code != 200:
                text = f"Error {response.status_code}:\n{response.text}"
            else:
                data = response.json()
                resultados = data.get("resultados", [])
                if not resultados:
                    text = "No se encontraron resultados."
                else:
                    partes = []
                    for i, r in enumerate(resultados, 1):
                        page = r.get("page")
                        partes.append(
                            f"--- Resultado {i} (página {page if page is not None else 'desconocida'}) ---\n"
                            f"{r.get('text', '')}\n"
                        )
                    text = "\n".join(partes)
        except Exception as e:
            text = f"Error al conectar con la API:\n{e}"

        self.root.after(
            0,
            lambda: (
                self.set_answer(text),
                self.set_buttons_state("normal"),
                self.spinner.config(text=""),
            ),
        )

    def clear_all(self):
        self.question_text.delete("1.0", tk.END)
        self.set_answer("")


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
