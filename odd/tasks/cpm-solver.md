# Soporte CPM/PERT en /resolver

## Objetivo
Extender el endpoint `/resolver` para que, además de Programación Lineal, pueda resolver problemas de CPM/PERT (ruta crítica) calculando matemáticamente:
- Duración total del proyecto.
- Ruta crítica.
- Holguras (slack) de cada actividad.
- Tiempos ES, EF, LS, LF de cada actividad.
- Todos los caminos posibles y sus duraciones.

## Problema
Hoy `/resolver` solo reconoce LP. Cuando recibe un problema de CPM/PERT, la IA lo clasifica como `OTRO` y solo da una explicación textual. El usuario quiere la solución numérica exacta.

## Solución propuesta
- Agregar en `solver.py` una función `resolver_cpm(actividades)` que implemente el algoritmo clásico de ruta crítica.
- Actualizar el prompt de `/resolver` para que la IA pueda devolver `"tipo": "CPM"` con la lista de actividades.
- En `/resolver`, si `tipo == "CPM"`, llamar a `resolver_cpm` en lugar de `resolver_lp_completo`.
- Incluir la salida CPM en formato matricial/tabular.
- Actualizar `gui.py` para mostrar resultados CPM de forma clara.

## Alcance
- Resolver CPM determinístico con duraciones fijas (no PERT probabilístico por ahora).
- Soportar predecesores múltiples por actividad.
- No modificar `.env` ni el comportamiento de `/buscar` y `/preguntar`.

## Tareas
1. [x] Implementar `resolver_cpm(actividades)` en `solver.py`.
2. [x] Implementar `formatear_cpm_matriz(resultado)` en `solver.py`.
3. [x] Actualizar el prompt de `/resolver` para reconocer CPM y devolver JSON estándar.
4. [x] En `/resolver`, rutear `tipo == "CPM"` al solver CPM.
5. [x] Actualizar `gui.py` para mostrar ruta crítica, holguras y tabla CPM.
6. [x] Verificar con el caso "Desarrollo de Software de Seguridad Informática".
7. [x] Verificar que un LP factible sigue funcionando igual.

## Formato JSON esperado de la IA para CPM
```json
{
  "tipo": "CPM",
  "actividades": [
    {"id": "A", "nombre": "Requerimientos", "duracion": 3, "predecesores": []},
    {"id": "B", "nombre": "Diseño BD", "duracion": 4, "predecesores": ["A"]},
    {"id": "C", "nombre": "Codificación", "duracion": 6, "predecesores": ["A"]},
    {"id": "D", "nombre": "Pruebas", "duracion": 2, "predecesores": ["B", "C"]}
  ],
  "explicacion": "..."
}
```

## Criterios de aceptación
- `POST /resolver` con un problema CPM devuelve `tipo_detectado: "CPM"`, duración total, ruta crítica, holguras y tabla ES/EF/LS/LF.
- El resultado del caso de prueba debe dar duración total 11, ruta crítica `[A, C, D]` y holgura de B = 2.
- `POST /resolver` con LP sigue funcionando igual.

## Checks
- [x] `python -c "import solver; print('ok')"`.
- [x] `python -c "import api; print('ok')"`.
- [x] Resolver CPM de ejemplo devuelve ruta crítica y holguras correctas.
- [x] Resolver LP de ejemplo sigue devolviendo solución óptima.

## Notas
- El repositorio no es Git; no se harán commits.
- Si la IA sigue clasificando CPM como OTRO, hay que ajustar el prompt hasta que sea robusto.
