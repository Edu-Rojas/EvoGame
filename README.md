# Guerra de especies

Ecosistema evolutivo: cada pana crea su especie repartiendo puntos entre genes, y las
criaturas comen, se aparean con herencia y mutación, envejecen y mueren. Nadie programa
quién gana: sale de los costos y beneficios de cada gen.

**Etapa actual: 1a** (herbívoros en un mundo con biomas, pasto que rebrota, reproducción sexual,
visor de desarrollo con animación procedural estilo Rain World).

## Requisitos

- Python 3.11 o más nuevo (usa `tomllib`, que viene con Python desde la 3.11)

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
| Ver el mundo en vivo | `python viewer/pygame_viewer.py` |
| Otra semilla / más rápido | `python viewer/pygame_viewer.py --seed 7 --tps 40` |
| Simulación sin gráficos | `python scripts/headless.py --ticks 5000 --every 500` |
| Tests | `pytest` |

Controles del visor:

| Tecla | Acción |
|---|---|
| Rueda del mouse | Zoom hacia el cursor (las patas y plumas se ven de cerca) |
| Clic derecho + arrastrar, o WASD | Mover la cámara |
| Clic izquierdo | Elegir criatura: ficha (genes, instintos, padres, energía) y radio de detección |
| F | La cámara sigue a la criatura elegida |
| Espacio | Pausa |
| ↑ / ↓ | Más o menos ticks por segundo |
| ESC | Salir |

## Ajustar el balance

Todos los números están en `config/default.toml`. Para experimentar, copia el archivo,
cámbialo y córrelo con `--config config/mi_prueba.toml`. Si escribes mal una clave, la
carga falla con un error claro en vez de ignorarla.

## Estructura

```
src/especies/      simulación pura (no sabe que existen los gráficos)
  config.py        TOML -> dataclasses inmutables, valida claves y presupuesto de genes
  genes.py         índices de genes, cruce uniforme, mutación, redondeo al azar
  state.py         World: criaturas como columnas de arrays + rasgos derivados de genes
  terrain.py       biomas (pradera, bosque, desierto, montaña, helada, agua) y pasto por celda
  perception.py    mejor celda de pasto + pareja más cercana (KD-tree en mundo toroidal)
  decision.py      utilidad por acción = instinto base x instinto heredable
  physics.py       toro, movimiento, deterioro por vejez, separación de cuerpos
  metabolism.py    pastar (reparto justo), gasto (Kleiber + ojos + movimiento + bioma), muerte
  reproduction.py  apareamiento, tamaño de camada, herencia
  step.py          orden de los sistemas en un tick
  metrics.py       resúmenes por especie y ficha de criatura
viewer/            visor pygame (solo lee el estado)
  animation.py     animación procedural: columna que sigue a la cabeza, patas con IK, plumas
  pygame_viewer.py cámara con zoom, terreno, dibujo, interpolación entre ticks
scripts/           headless
tests/             pytest
```

## Qué hay en la 1a y qué falta

- Activos: Tamaño, Aceleración (solo velocidad, sin fatiga), Detección, Apareamiento, Dieta
  (todos herbívoros).
- Agresividad y Camuflaje existen y se heredan, pero todavía no hacen nada: van a derivar
  al azar (deriva neutral), lo cual es interesante de ver.
- En la 1a el Tamaño casi solo tiene costos (su única ventaja todavía es aguantar el frío de
  la helada, por la regla de Bergmann; pelear y cazar llegan en la 1b), así que la selección
  lo empuja hacia abajo y a los grandes les va mal. No es un bug.
- Biomas: cambian cuánto pasto crece, la velocidad, cuánto se ve (en el bosque menos) y el
  gasto de energía. El agua se cruza nadando, lento.
- Sin stats todavía (Percepción, Ataque...): detectar dentro del radio es seguro.
- **1b:** carnívoros, cazar, huir, peleas, cadáveres y carroña.
- **1c:** camuflaje, esconderse/acechar, fatiga de Aceleración.
