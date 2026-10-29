"""Solvers numéricos para problemas de Investigación de Operaciones.

Expone funciones para resolver problemas de programación lineal usando scipy
(u otro solver) de forma que la API pueda obtener resultados exactos.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


CODIGO_ESTADO_OPTIMO = 0
CODIGO_ESTADO_INFACTIBLE = 2
CODIGO_ESTADO_ILIMITADO = 3


def resolver_lp(
    c: List[float],
    objetivo: str = "minimizar",
    A_ub: Optional[List[List[float]]] = None,
    b_ub: Optional[List[float]] = None,
    A_lb: Optional[List[List[float]]] = None,
    b_lb: Optional[List[float]] = None,
    A_eq: Optional[List[List[float]]] = None,
    b_eq: Optional[List[float]] = None,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
) -> Dict[str, Any]:
    """Resuelve un problema de programación lineal con scipy.optimize.linprog.

    Parámetros:
        c: coeficientes de la función objetivo.
        objetivo: "minimizar" o "maximizar".
        A_ub, b_ub: restricciones de la forma A_ub @ x <= b_ub.
        A_lb, b_lb: restricciones de la forma A_lb @ x >= b_lb.
                      El solver las convierte internamente a <= multiplicando por -1.
        A_eq, b_eq: restricciones de igualdad (A_eq @ x == b_eq).
        bounds: límites por variable como lista de (min, max). None significa sin límite.

    Retorna un diccionario con:
        - status: "ÓPTIMO", "NO_ÓPTIMO", "ERROR".
        - codigo_estado: código numérico de HiGHS/scipy (0=óptimo, 2=infactible,
          3=ilimitado, etc.; -1 si hubo una excepción).
        - solucion: lista de valores de x (solo si es óptimo).
        - valor_objetivo: valor óptimo de Z.
        - message: mensaje del solver o del error.
        - funcion_objetivo_usada: coeficientes reales enviados al solver.
    """
    try:
        c_array = np.array(c, dtype=float)

        if objetivo.lower() == "maximizar":
            c_solver = -c_array
        else:
            c_solver = c_array

        # Convertir restricciones >= a <= de forma robusta
        A_ub_final = A_ub if A_ub is not None else []
        b_ub_final = b_ub if b_ub is not None else []

        if A_lb and b_lb:
            A_lb_array = np.array(A_lb, dtype=float)
            b_lb_array = np.array(b_lb, dtype=float)
            A_ub_final = np.vstack([A_ub_final, -A_lb_array]) if len(A_ub_final) else -A_lb_array
            b_ub_final = np.concatenate([b_ub_final, -b_lb_array]) if len(b_ub_final) else -b_lb_array

        if len(A_ub_final) == 0:
            A_ub_final = None
            b_ub_final = None

        result = linprog(
            c=c_solver,
            A_ub=A_ub_final,
            b_ub=b_ub_final,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if result.success:
            valor = result.fun
            if objetivo.lower() == "maximizar":
                valor = -valor
            return {
                "status": "ÓPTIMO",
                "codigo_estado": int(result.status),
                "solucion": result.x.tolist(),
                "valor_objetivo": float(valor),
                "message": result.message,
                "funcion_objetivo_usada": c_array.tolist(),
            }

        return {
            "status": "NO_ÓPTIMO",
            "codigo_estado": int(result.status),
            "solucion": None,
            "valor_objetivo": None,
            "message": result.message,
            "funcion_objetivo_usada": c_array.tolist(),
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "codigo_estado": -1,
            "solucion": None,
            "valor_objetivo": None,
            "message": str(e),
            "funcion_objetivo_usada": c,
        }


def inferir_bounds(n: int, bounds: Optional[List] = None) -> List[Tuple[Optional[float], Optional[float]]]:
    """Devuelve bounds por defecto no negativas si no se especifican."""
    if bounds is None:
        return [(0, None)] * n
    # Completar con (0, None) si faltan
    resultado = []
    for b in bounds:
        if b is None:
            resultado.append((0, None))
        else:
            min_val = b[0] if b[0] is not None else 0.0
            max_val = b[1] if b[1] is not None else None
            resultado.append((min_val, max_val))
    # Rellenar el resto
    while len(resultado) < n:
        resultado.append((0, None))
    return resultado


def _limpiar_restricciones(A, b):
    """Devuelve None si la matriz o vector están vacíos, para evitar errores de scipy."""
    if not A or not b:
        return None, None
    return A, b


def resolver_lp_completo(datos: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper que normaliza bounds y delega a resolver_lp.

    Espera un diccionario con las mismas claves que resolver_lp.
    """
    c = datos.get("c", [])
    n = len(c)
    bounds_raw = datos.get("bounds")
    bounds = inferir_bounds(n, bounds_raw) if bounds_raw is not None else [(0, None)] * n

    A_ub, b_ub = _limpiar_restricciones(datos.get("A_ub"), datos.get("b_ub"))
    A_lb, b_lb = _limpiar_restricciones(datos.get("A_lb"), datos.get("b_lb"))
    A_eq, b_eq = _limpiar_restricciones(datos.get("A_eq"), datos.get("b_eq"))

    return resolver_lp(
        c=c,
        objetivo=datos.get("objetivo", "minimizar"),
        A_ub=A_ub,
        b_ub=b_ub,
        A_lb=A_lb,
        b_lb=b_lb,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
    )


def es_infactible(solucion: dict) -> bool:
    """Devuelve True si la solución reporta infactibilidad según HiGHS."""
    if not isinstance(solucion, dict):
        return False
    return solucion.get("codigo_estado") == CODIGO_ESTADO_INFACTIBLE


def aplicar_cambios_modelo(datos: dict, cambios: list) -> dict:
    """Aplica cambios conservadores sobre una copia profunda del modelo LP.

    Formatos soportados para cada cambio:
        - {"tipo": "bound", "variable": int, "min": ..., "max": ...}
        - {"tipo": "restriccion_ub", "indice": int, "nuevo_b": ...}
        - {"tipo": "restriccion_lb", "indice": int, "nuevo_b": ...}
        - {"tipo": "restriccion_eq", "indice": int, "nuevo_b": ...}

    Los índices fuera de rango se ignoran o se clampan al rango válido.
    """
    modelo = copy.deepcopy(datos)
    n = len(modelo.get("c", []))

    # Asegurar que existan bounds coherentes si el modelo los va a necesitar.
    if modelo.get("bounds") is None and n > 0:
        modelo["bounds"] = [[0, None] for _ in range(n)]

    if not isinstance(cambios, list):
        return modelo

    for cambio in cambios:
        if not isinstance(cambio, dict):
            continue

        tipo = cambio.get("tipo")

        if tipo == "bound":
            idx = cambio.get("variable")
            if not isinstance(idx, int):
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    continue
            if modelo.get("bounds") is None or n == 0:
                continue
            if idx < 0 or idx >= n:
                idx = max(0, min(idx, n - 1))
            bound = modelo["bounds"][idx]
            if bound is None:
                bound = [0, None]
                modelo["bounds"][idx] = bound
            if "min" in cambio:
                bound[0] = cambio["min"]
            if "max" in cambio:
                bound[1] = cambio["max"]

        elif tipo in ("restriccion_ub", "restriccion_lb", "restriccion_eq"):
            idx = cambio.get("indice")
            if not isinstance(idx, int):
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    continue
            if idx < 0:
                continue

            clave_b = {
                "restriccion_ub": "b_ub",
                "restriccion_lb": "b_lb",
                "restriccion_eq": "b_eq",
            }[tipo]

            b = modelo.get(clave_b)
            if not isinstance(b, list) or idx >= len(b):
                continue
            b[idx] = cambio.get("nuevo_b", b[idx])

    return modelo


def resolver_cpm(actividades: list) -> dict:
    """Resuelve un problema de CPM/PERT determinístico por ruta crítica.

    Parámetros:
        actividades: lista de diccionarios con claves:
            - id: identificador único (str).
            - nombre: descripción legible (str, opcional).
            - duracion: duración numérica de la actividad.
            - predecesores: lista de ids de actividades predecessoras.

    Retorna un diccionario con:
        - duracion_total: duración máxima del proyecto.
        - ruta_critica: lista ordenada de ids que forman el camino crítico.
        - actividades: lista con ES, EF, LS, LF, holgura y bandera critica.
        - caminos: lista de todos los caminos posibles con su duración.
    """
    if not isinstance(actividades, list):
        raise ValueError("actividades debe ser una lista")

    act_map = {}
    for act in actividades:
        if not isinstance(act, dict) or "id" not in act:
            raise ValueError("Cada actividad debe ser un diccionario con 'id'")
        act_id = act["id"]
        if act_id in act_map:
            raise ValueError(f"Actividad duplicada: {act_id}")
        act_map[act_id] = {
            "id": act_id,
            "nombre": act.get("nombre", act_id),
            "duracion": float(act.get("duracion", 0)),
            "predecesores": list(act.get("predecesores", [])),
            "sucesores": [],
            "ES": 0.0,
            "EF": 0.0,
            "LS": 0.0,
            "LF": 0.0,
            "holgura": 0.0,
            "critica": False,
        }

    # Validar predecesores y construir sucesores
    for act in act_map.values():
        for pred in act["predecesores"]:
            if pred not in act_map:
                raise ValueError(
                    f"Predecesor '{pred}' no encontrado para actividad '{act['id']}'"
                )
            act_map[pred]["sucesores"].append(act["id"])

    # Forward pass con orden topológico (Kahn)
    in_degree = {aid: len(act_map[aid]["predecesores"]) for aid in act_map}
    queue = [aid for aid, deg in in_degree.items() if deg == 0]
    topo_order = []

    while queue:
        current_id = queue.pop(0)
        topo_order.append(current_id)
        current = act_map[current_id]

        if not current["predecesores"]:
            current["ES"] = 0.0
        else:
            current["ES"] = max(act_map[pred]["EF"] for pred in current["predecesores"])
        current["EF"] = current["ES"] + current["duracion"]

        for succ in current["sucesores"]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(topo_order) != len(act_map):
        raise ValueError("Se detectó un ciclo en las dependencias de las actividades")

    # Duración total del proyecto
    finales = [aid for aid in act_map if not act_map[aid]["sucesores"]]
    duracion_total = (
        max(act_map[aid]["EF"] for aid in finales) if finales else 0.0
    )

    # Backward pass en orden topológico inverso
    for aid in reversed(topo_order):
        act = act_map[aid]
        if not act["sucesores"]:
            act["LF"] = duracion_total
        else:
            act["LF"] = min(act_map[succ]["LS"] for succ in act["sucesores"])
        act["LS"] = act["LF"] - act["duracion"]
        act["holgura"] = act["LS"] - act["ES"]
        act["critica"] = abs(act["holgura"]) < 1e-9

    # Calcular todos los caminos posibles de inicio a fin
    iniciales = [aid for aid in act_map if not act_map[aid]["predecesores"]]
    caminos = []

    def dfs(camino_actual: list, id_actual):
        camino_actual.append(id_actual)
        act = act_map[id_actual]
        if not act["sucesores"]:
            duracion = sum(act_map[a]["duracion"] for a in camino_actual)
            caminos.append({"camino": list(camino_actual), "duracion": duracion})
        else:
            for succ in act["sucesores"]:
                dfs(camino_actual, succ)
        camino_actual.pop()

    for inicial in iniciales:
        dfs([], inicial)

    # Ruta crítica: camino de mayor duración
    ruta_critica = []
    max_duracion = -1.0
    for camino_info in caminos:
        if camino_info["duracion"] > max_duracion:
            max_duracion = camino_info["duracion"]
            ruta_critica = list(camino_info["camino"])

    # Preparar lista de actividades en orden topológico
    actividades_resultado = []
    for aid in topo_order:
        act = act_map[aid]
        actividades_resultado.append(
            {
                "id": act["id"],
                "nombre": act["nombre"],
                "duracion": act["duracion"],
                "ES": act["ES"],
                "EF": act["EF"],
                "LS": act["LS"],
                "LF": act["LF"],
                "holgura": act["holgura"],
                "critica": act["critica"],
            }
        )

    return {
        "duracion_total": duracion_total,
        "ruta_critica": ruta_critica,
        "actividades": actividades_resultado,
        "caminos": caminos,
    }


def formatear_cpm_matriz(resultado: dict) -> dict:
    """Devuelve el resultado CPM en formato matricial/tabular."""
    if not isinstance(resultado, dict):
        return {}

    tabla_actividades = []
    for act in resultado.get("actividades", []):
        tabla_actividades.append(
            {
                "id": act.get("id"),
                "nombre": act.get("nombre"),
                "duracion": _valor_celda(act.get("duracion")),
                "ES": _valor_celda(act.get("ES")),
                "EF": _valor_celda(act.get("EF")),
                "LS": _valor_celda(act.get("LS")),
                "LF": _valor_celda(act.get("LF")),
                "holgura": _valor_celda(act.get("holgura")),
                "critica": "Si" if act.get("critica") else "No",
            }
        )

    return {
        "tabla_actividades": tabla_actividades,
        "ruta_critica": resultado.get("ruta_critica", []),
        "duracion_total": _valor_celda(resultado.get("duracion_total")),
        "caminos": resultado.get("caminos", []),
    }


def _nombre_variable(i: int) -> str:
    return f"x{i + 1}"


def _valor_celda(v):
    """Normaliza un valor numérico para JSON: None -> None (se representa como null)."""
    if v is None:
        return None
    return float(v)


def formatear_modelo_matriz(datos: dict, solucion: Optional[dict] = None) -> dict:
    """Devuelve el modelo LP y su solución en formato matricial tipo Excel.

    La salida incluye tablas con columnas por variable, tipo de restricción,
    RHS, bounds y valores de la solución. Es fácil de renderizar en una GUI o
    exportar a CSV/Excel.
    """
    n = len(datos.get("c", []))
    nombres = [_nombre_variable(i) for i in range(n)]

    # Función objetivo como fila
    fila_objetivo = {"Z": datos.get("objetivo", "minimizar")}
    for i, nombre in enumerate(nombres):
        fila_objetivo[nombre] = _valor_celda(datos.get("c", [])[i] if i < n else None)

    # Restricciones como filas de matriz
    restricciones = []

    def _agregar_filas(A, b, simbolo):
        if not A or not b:
            return
        for idx, (fila, rhs) in enumerate(zip(A, b)):
            fila_dict = {
                "restriccion": f"R{len(restricciones) + 1}",
                "tipo": simbolo,
                "RHS": _valor_celda(rhs),
            }
            for j, nombre in enumerate(nombres):
                fila_dict[nombre] = _valor_celda(fila[j] if j < len(fila) else None)
            restricciones.append(fila_dict)

    _agregar_filas(datos.get("A_ub"), datos.get("b_ub"), "<=")
    _agregar_filas(datos.get("A_lb"), datos.get("b_lb"), ">=")
    _agregar_filas(datos.get("A_eq"), datos.get("b_eq"), "=")

    # Bounds como tabla
    bounds_raw = datos.get("bounds")
    bounds = inferir_bounds(n, bounds_raw) if bounds_raw is not None else [(0, None)] * n
    tabla_bounds = []
    for i, nombre in enumerate(nombres):
        min_v, max_v = bounds[i] if i < len(bounds) else (0, None)
        tabla_bounds.append(
            {
                "variable": nombre,
                "min": _valor_celda(min_v),
                "max": _valor_celda(max_v),
            }
        )

    # Solución
    solucion_tabla = []
    valor_z = None
    if solucion:
        valores = solucion.get("solucion")
        if valores:
            for i, nombre in enumerate(nombres):
                solucion_tabla.append(
                    {
                        "variable": nombre,
                        "valor": _valor_celda(valores[i] if i < len(valores) else None),
                    }
                )
        valor_z = _valor_celda(solucion.get("valor_objetivo"))

    return {
        "funcion_objetivo": fila_objetivo,
        "restricciones": restricciones,
        "bounds": tabla_bounds,
        "solucion": {
            "variables": solucion_tabla,
            "valor_objetivo": valor_z,
            "status": solucion.get("status") if solucion else None,
        },
    }
