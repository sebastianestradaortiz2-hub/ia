# Reparación automática de problemas infactibles en /resolver

## Objetivo
Cuando el endpoint `/resolver` detecte que un modelo de Programación Lineal no tiene solución por infactibilidad, debe:
1. Consultar la base vectorial por teoría de factibilidad y análisis de sensibilidad.
2. Pedir a la IA que proponga cambios concretos sobre restricciones y/o bounds para hacerlo factible.
3. Re-solver el modelo modificado numéricamente.
4. Responder al usuario con: "Haciendo estos cambios el problema SÍ es factible", mostrando qué cambió, por qué y la nueva solución.

## Problema
Hoy `/resolver` devuelve `status: "NO_ÓPTIMO"` cuando el solver encuentra infactibilidad, pero no ayuda al usuario a entender qué ajustes harían el problema viable. Eso deja al usuario sin una acción concreta.

## Solución propuesta
- Detectar código de estado `2` (infactible) de `scipy.optimize.linprog`.
- Buscar contexto adicional en la base vectorial con queries como:
  - "infactibilidad programación lineal"
  - "análisis de sensibilidad restricciones"
  - "hacer factible un modelo de programación lineal"
- Enviar un segundo prompt a OpenRouter con:
  - El modelo LP original.
  - El mensaje de infactibilidad del solver.
  - Contexto teórico del libro.
  - Pedido de JSON con cambios propuestos (bounds y/o RHS de restricciones).
- Aplicar los cambios al diccionario del modelo y re-solver.
- Incluir en la respuesta un campo `propuesta_factibilidad` con los cambios, justificación y nueva solución.

## Alcance
- Modificar `solver.py` para exponer el código de estado HiGHS y una función que aplique cambios al modelo.
- Modificar `api.py` endpoint `/resolver` para la lógica de reparación.
- Actualizar `gui.py` para mostrar la propuesta cuando exista.
- No tocar `.env` ni el comportamiento base de `/buscar` y `/preguntar`.

## Tareas
1. [x] Extender `resolver_lp` para retornar `codigo_estado` de HiGHS.
2. [x] Agregar helper `es_infactible(solucion)` en `solver.py`.
3. [x] Agregar `aplicar_cambios_modelo(datos, cambios)` en `solver.py` (soportar bounds, b_ub, b_lb, b_eq).
4. [x] En `/resolver`, después de resolver, si es infactible:
   - [x] Buscar contexto sobre factibilidad/análisis de sensibilidad.
   - [x] Llamar a OpenRouter con prompt de reparación.
   - [x] Parsear JSON de cambios propuestos.
   - [x] Aplicar cambios y re-solver.
   - [x] Armar respuesta con `propuesta_factibilidad`.
5. [x] Actualizar `gui.py` para renderizar propuesta y nueva solución.
6. [x] Agregar formato matricial tipo Excel a la salida numérica del solver.
7. [x] Probar con un LP infactible conocido.
8. [x] Probar que un LP factible sigue funcionando igual.

## Criterios de aceptación
- `/resolver` con un LP infactible devuelve `propuesta_factibilidad` con cambios, justificación y una solución reparada factible.
- Si la propuesta reparada sigue siendo infactible, se informa honestamente.
- `/resolver` con LP factible no cambia de comportamiento.
- `gui.py` muestra claramente la propuesta de factibilidad.

## Variables de entorno
- Sin cambios. Usa `OPENROUTER_API_KEY` existente.

## Checks
- [x] `python -c "import solver; print('ok')"`.
- [x] `python -c "import api; print('ok')"` (con timeout largo por carga de modelos).
- [x] `python -c "import gui; print('ok')"`.
- [x] Petición infactible a `/resolver` devuelve propuesta y solución reparada.
- [x] Petición factible a `/resolver` devuelve solución directa.

## Notas
- El repositorio no es Git; no se harán commits.
- Los cambios propuestos deben ser conservadores: preferir relajar RHS o bounds antes que alterar la estructura del problema.
