# Guerra de especies

Ecosistema evolutivo: cada jugador crea una especie repartiendo puntos entre genes, y las
criaturas comen, se aparean con herencia y mutación, envejecen y mueren. Nadie programa
quién gana: sale de los costos y beneficios de cada gen.

**Etapa actual: 1b.** Herbívoros y carnívoros en un mundo con biomas: caza, huida,
peleas, carne que se pudre y hojas altas que solo alcanzan los grandes. Visor de
desarrollo en pixel art con animación procedural.

## Requisitos

- Python 3.11 o más nuevo (usa `tomllib`)

## Instalación (Windows)

```powershell
cd guerra-de-especies
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[viewer,dev]"
```

En Linux o macOS, `source .venv/bin/activate` en vez de la segunda línea.

## Cómo correrlo

| Qué | Comando |
|---|---|
| Ver el mundo en vivo | `python -m viewer` |
| Otra semilla / más rápido | `python -m viewer --seed 7 --tps 40` |
| Simulación sin gráficos | `especies-headless --ticks 5000 --every 500` |
| Aceptación (varias semillas en paralelo) | `especies-acceptance --ticks 50000 --out resultado.md` |
| Tests (todos / rápidos) | `pytest` / `pytest -m "not slow"` |
| Lint y tipos | `ruff check .` y `mypy` |

Controles del visor:

| Tecla | Acción |
|---|---|
| Rueda del mouse | Zoom hacia el cursor (de cerca se ven patas, garras y branquias) |
| Clic derecho + arrastrar, o WASD | Mover la cámara |
| Clic izquierdo | Elegir criatura: ficha (genes, instintos, padres, energía) y radio de detección |
| F | La cámara sigue a la criatura elegida |
| P | Tamaño del píxel (2, 3 o 4) |
| Espacio | Pausa |
| ↑ / ↓ | Más o menos ticks por segundo |
| ESC | Salir |

## Ajustar el balance

Todos los números de balance están en `src/especies/data/default.toml`, que viaja dentro
del paquete. Para experimentar, copia el archivo, cámbialo y córrelo con
`--config mi_prueba.toml`. Si escribes mal una clave o pones un valor fuera de rango, la
carga falla con un error que dice la sección y la clave.

## Estructura

```
src/especies/        simulación pura (no sabe que existen los gráficos)
  config.py          TOML -> dataclasses inmutables; valida claves, rangos y presupuesto
  data/default.toml  todos los números de balance
  cli.py             especies-headless: reportes por especie sin gráficos
  geometry.py        mundo toroidal: wrap y camino más corto
  genes.py           genes, cruce uniforme, mutación, redondeo al azar
  state.py           World: criaturas como columnas de arrays + rasgos derivados
  terrain.py         biomas; pasto, hojas altas y carne por celda
  perception.py      comida, pareja, presa y amenaza (KD-trees por especie)
  decision.py        utilidad por acción = instinto base x instinto heredable
  physics.py         movimiento, huida hacia la cobertura, filopatría, separación
  combat.py          mordidas, devolución del golpe, regeneración
  metabolism.py      comer, gasto (Kleiber + ojos + movimiento + bioma), muerte y carne
  disease.py         enfermedad por densidad
  reproduction.py    apareamiento, tamaño de camada, herencia
  step.py            orden de los sistemas en un tick
  metrics.py         resúmenes por especie y ficha de criatura (datos crudos)
  acceptance.py      especies-acceptance: corridas largas con varias semillas
viewer/              visor pygame (solo lee el estado)
  animation.py       rig procedural en numpy: columna, patas con IK, garras, branquias
  palette.py         tinta, papel y colores por especie
  render.py          pixel art: terreno con dithering y garabatos, criaturas con LOD
  app.py             cámara, entrada, interpolación entre ticks y HUD
tests/               pytest, un archivo por módulo
```

## Documentación

- [CHANGELOG.md](CHANGELOG.md): cambios por versión.

## Qué hay hasta la 1b y qué falta

- Activos: Tamaño, Aceleración (solo velocidad, sin fatiga), Agresividad, Dieta,
  Detección y Apareamiento. Camuflaje se hereda pero todavía no muta
  (`[genes] inactive`): se activa en la 1c.
- Carnívoros: cazan presas de otra especie hasta `max_prey_ratio` veces su tamaño
  (refugio por tamaño), se sacian, se rinden si la persecución es larga y dejan carne.
- Hojas altas en el bosque: el nicho propio de los grandes.
- Coexistencia: sin mecanismos, la especie que más se reproduce excluye a las demás en
  una generación. Las mecánicas que la estabilizan están en
  [tests/test_stabilizers.py](tests/test_stabilizers.py). Hoy, en 4 de 5 semillas
  sobreviven al menos 3 especies durante 50.000 ticks.
- **1c:** camuflaje, esconderse/acechar, fatiga de Aceleración.
