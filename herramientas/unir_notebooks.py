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
# libreria de expresiones regulares para reconocer el marcador de inclusion
import re

# raiz del repo (carpeta superior a herramientas)
RAIZ = Path(__file__).resolve().parent.parent
# orden de union: parte 1, parte 2 (filtros) y parte 2 (CNN)
ORDEN = ["Parte_A_Fourier.ipynb", "parte2_fft_filtros.ipynb", "parte2_cnn_comparacion.ipynb"]
# marcador que indica que la celda debe sustituirse por un archivo (acepta "# @INCLUIR:" y "#@INCLUIR:")
MARCADOR = re.compile(r"#\s*@INCLUIR:\s*(\S+)")
# importaciones desde src/ que deben omitirse en el notebook final (el nucleo ya queda incluido en una celda)
IMPORTACION = re.compile(r"^(\s*)from fft_nucleo import .*$", re.MULTILINE)


# librerias que necesita el notebook final (la primera celda las instala si faltan)
PIP_INSTALAR = "numpy matplotlib pillow scipy pandas scikit-image ipywidgets torch"
# texto de la celda markdown inicial
TEXTO_INSTALACION = (
    "# MAP Matematica 6: Serie de Fourier truncada y FFT\n\n"
    "**Primera celda: instalacion de librerias.** Ejecutela una vez; si ya estan instaladas no hace nada. "
    "Si se instala algo nuevo (por ejemplo `ipywidgets`), reinicie el kernel y ejecute el notebook desde el inicio."
)
# codigo de la celda de instalacion
CODIGO_INSTALACION = (
    "# instalacion de todas las librerias necesarias en el Python del kernel actual\n"
    "%pip install -q " + PIP_INSTALAR
)


# funcion que construye el notebook final
def unir():
    # notebook nuevo y vacio
    final = nbformat.v4.new_notebook()
    # kernel de Python 3
    final.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    # las dos primeras celdas: explicacion e instalacion de librerias
    final.cells.append(nbformat.v4.new_markdown_cell(TEXTO_INSTALACION))
    final.cells.append(nbformat.v4.new_code_cell(CODIGO_INSTALACION))
    # archivos ya incluidos, para no repetir el nucleo dos veces
    incluidos = set()
    # recorrido de cada notebook en orden
    for nombre in ORDEN:
        # lectura del notebook de un integrante
        nb = nbformat.read(RAIZ / "notebooks" / nombre, as_version=4)
        # recorrido de sus celdas
        for celda in nb.cells:
            # busqueda del marcador de inclusion al inicio de las celdas de codigo
            coincidencia = MARCADOR.match(celda.source) if celda.cell_type == "code" else None
            # celdas con marcador de inclusion
            if coincidencia:
                # ruta del archivo a incluir (texto que sigue al marcador)
                ruta = coincidencia.group(1)
                # el nucleo se incluye una sola vez
                if ruta in incluidos:
                    continue
                # se registra como incluido
                incluidos.add(ruta)
                # la celda pasa a contener el codigo del archivo
                celda = nbformat.v4.new_code_cell((RAIZ / ruta).read_text(encoding="utf-8"))
            # en las celdas de codigo se omiten las importaciones desde src/ para que el notebook sea autonomo
            if celda.cell_type == "code":
                celda.source = IMPORTACION.sub(r"\1pass  # importacion omitida: el nucleo ya esta incluido arriba", celda.source)
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
