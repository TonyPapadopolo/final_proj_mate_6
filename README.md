# MAP Matematica 6: Serie de Fourier Truncada y FFT

Proyecto en grupo (Christian, Tony, Claudia). Entrega: **ZIP en GES** con `.tex`, imagenes del `.tex`, PDF del reporte y el notebook de ambas partes. Presentacion virtual el 7 de noviembre. **Confirmar si la fecha de subida a GES es el 7 u 8** (el PDF dice domingo 8).

## Reparto
| Quien | Que | Archivo de trabajo |
|---|---|---|
| Christian (A) | Parte 1 completa: series, ICE, interfaz, funcion de IA | `notebooks/parte1_series_fourier.ipynb` |
| Tony (B) | Parte 2 incisos 1 a 3: FFT propia, pasa bajas, pasa altas, Blur Detection | `src/fft_nucleo.py` y `notebooks/parte2_fft_filtros.ipynb` |
| Claudia (C) | Parte 2 incisos 4 y 5: CNN, ruido, comparacion; estructura de LaTeX y diapositivas | `notebooks/parte2_cnn_comparacion.ipynb` y `reporte/main.tex` |

## Estructura
```
src/fft_nucleo.py        FFT recursiva 1D/2D, filtros, Blur Detection (unica fuente del codigo)
notebooks/               un notebook por integrante (evita conflictos de git en los .ipynb)
herramientas/            unir_notebooks.py: genera notebooks/MAP_final.ipynb para la entrega
imagenes/                imagenes del proyecto (si esta vacia se usa una sintetica de respaldo)
reporte/main.tex         reporte en LaTeX, una seccion por persona
reporte/figuras/         figuras generadas por los notebooks (savefig dpi=200)
```

## Como correr
```
pip install -r requirements.txt
cd notebooks            # los notebooks usan rutas relativas desde esta carpeta
jupyter lab             # abrir y ejecutar el notebook propio (Restart & Run All)
```
Para la entrega, desde la raiz del repo: `python herramientas/unir_notebooks.py`. Genera `notebooks/MAP_final.ipynb` autonomo (incluye el codigo de `src/` en una celda). Ejecutarlo completo en limpio antes de subirlo.

## Reglas para no pisarse en git
- Cada quien edita **solo su notebook y su seccion** del `.tex`. Lo compartido (`src/fft_nucleo.py`) lo cambia Tony; si alguien necesita un cambio, que lo pida.
- Nombres de variables distintos entre partes si se llegan a juntar celdas (el notebook final usa un solo kernel).
- Capturas de **cada prompt de IA** en el momento (el reporte las exige).

## Convencion de frecuencia de corte (declarar en el reporte)
Radio normalizado `r` en [0, 1] desde el centro del espectro centrado: `r = 0` es la frecuencia cero y `r = 1` la esquina. Pasa bajas conserva `r <= fc`, pasa altas conserva `r > fc`. La FFT propia necesita lados potencia de 2, por eso las imagenes se ajustan a 256x256.

## Estado (25 de octubre)
- [x] FFT recursiva 1D, inversa, FFT 2D e inversa (probadas contra `np.fft`)
- [x] Pasa bajas, pasa altas (ideal y gaussiano), Blur Detection, figuras para el reporte
- [ ] Imagenes oficiales del proyecto en `imagenes/` y recalibrar `UMBRAL` del Blur Detection
- [ ] Parte 1 (Christian) y CNN (Claudia): solo esqueletos
