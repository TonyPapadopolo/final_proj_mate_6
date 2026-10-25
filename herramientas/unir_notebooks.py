# ======================================================================
# Une los notebooks de cada integrante en UN solo notebook de entrega.
# Uso (desde la raiz del repo):  python herramientas/unir_notebooks.py
# Resultado: notebooks/MAP_final.ipynb
# La celda que empieza con "# @INCLUIR: ruta" se reemplaza por el codigo de ese archivo,
# asi el notebook final es autonomo (no depende de src/).
# ======================================================================

# libreria para leer y escribir notebooks
import nbformat
# libreria para manejar rutas
from pathlib import Path

# raiz del repo (carpeta superior a herramientas)
RAIZ = Path(__file__).resolve().parent.parent
# orden de union: parte 1, parte 2 (filtros) y parte 2 (CNN)
ORDEN = ["parte1_series_fourier.ipynb", "parte2_fft_filtros.ipynb", "parte2_cnn_comparacion.ipynb"]
# marcador que indica que la celda debe sustituirse por un archivo
MARCADOR = "# @INCLUIR:"


# funcion que construye el notebook final
def unir():
    # notebook nuevo y vacio
    final = nbformat.v4.new_notebook()
    # kernel de Python 3
    final.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    # archivos ya incluidos, para no repetir el nucleo dos veces
    incluidos = set()
    # recorrido de cada notebook en orden
    for nombre in ORDEN:
        # lectura del notebook de un integrante
        nb = nbformat.read(RAIZ / "notebooks" / nombre, as_version=4)
        # recorrido de sus celdas
        for celda in nb.cells:
            # celdas con marcador de inclusion
            if celda.cell_type == "code" and celda.source.startswith(MARCADOR):
                # ruta del archivo a incluir (primera linea despues del marcador)
                ruta = celda.source.splitlines()[0].replace(MARCADOR, "").strip()
                # el nucleo se incluye una sola vez
                if ruta in incluidos:
                    continue
                # se registra como incluido
                incluidos.add(ruta)
                # la celda pasa a contener el codigo del archivo
                celda = nbformat.v4.new_code_cell((RAIZ / ruta).read_text(encoding="utf-8"))
            # las celdas repetidas de configuracion/carga se conservan; el resto se copia tal cual
            final.cells.append(celda)
    # escritura del notebook final
    destino = RAIZ / "notebooks" / "MAP_final.ipynb"
    nbformat.write(final, destino)
    # mensaje de confirmacion
    print(f"Notebook final generado: {destino}  ({len(final.cells)} celdas)")


# ejecucion al correr el script
if __name__ == "__main__":
    unir()
