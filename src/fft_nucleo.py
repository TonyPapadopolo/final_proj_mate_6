# ======================================================================
# NUCLEO DE LA PARTE 2: FFT (Cooley-Tukey), FFT 2D, filtros y Blur Detection
# Responsable: Tony (B). Este archivo es la unica fuente del codigo de la FFT.
# Regla del proyecto: la FFT implementada NO usa np.fft.fft ni np.fft.fft2.
# (np.fft solo se usa en las pruebas, para comparar, y np.fft.fftshift
#  solo para reordenar el espectro, sin calcular ninguna transformada.)
# ======================================================================

# libreria numerica
import numpy as np
# libreria para cargar y redimensionar imagenes
from PIL import Image
# libreria para listar archivos de una carpeta
from pathlib import Path


# ---------------------------------------------------------------------
# 1. FFT DE COOLEY-TUKEY (1D)
# ---------------------------------------------------------------------

# funcion que verifica si un numero entero es potencia de 2
def es_potencia_de_2(n):
    # una potencia de 2 tiene un solo bit en 1, por eso n & (n-1) vale 0
    return n >= 1 and (n & (n - 1)) == 0


# funcion que implementa el algoritmo FFT de Cooley-Tukey propuesto
def fft_recursiva(x):

    # conversion de la entrada a valores complejos
    x = np.asarray(x, dtype=complex)

    # N es la longitud del ultimo eje; asi la misma funcion transforma un vector (N,) o muchas filas a la vez (F, N) sin cambiar el algoritmo
    N = x.shape[-1]

    # el algoritmo solo funciona si N es potencia de 2
    if not es_potencia_de_2(N):
        
        # aviso claro del error para el usuario
        raise ValueError(f"La longitud debe ser potencia de 2, se recibio N={N}")
    
    # caso base del algoritmo: if N = 1 then return x
    if N == 1:

        # la DFT de un solo elemento es el mismo elemento
        return x
    
    # DFT recursiva de los indices pares: x0, x2, ..., x(N-2)
    x_par = fft_recursiva(x[..., 0::2])

    # DFT recursiva de los indices impares: x1, x3, ..., x(N-1)
    x_impar = fft_recursiva(x[..., 1::2])

    # indices k = 0, 1, ..., N/2 - 1
    k = np.arange(N // 2)

    # factor T = x_impar[k] * e^(-2*pi*i*k/N), calculado para todos los k a la vez
    T = x_impar * np.exp(-2j * np.pi * k / N)

    # X[k] = x_par[k] + T  y  X[k + N/2] = x_par[k] - T, unidos en un solo arreglo
    return np.concatenate([x_par + T, x_par - T], axis=-1)


# funcion que calcula la inversa usando la propiedad: IFFT(X) = conj(FFT(conj(X))) / N
def ifft_recursiva(X):

    # conversion de la entrada a valores complejos
    X = np.asarray(X, dtype=complex)

    # longitud de la secuencia
    N = X.shape[-1]

    # conjugar, aplicar la FFT, conjugar de nuevo y dividir entre N
    return np.conj(fft_recursiva(np.conj(X))) / N


# funcion de referencia: DFT directa O(N^2) con la formula del enunciado
def dft_directa(x):

    # conversion de la entrada a valores complejos
    x = np.asarray(x, dtype=complex)

    # cantidad de muestras
    N = x.size

    # indices n (columnas) y k (filas)
    n = np.arange(N)
    k = n.reshape((N, 1))

    # X[k] = suma de x[n] * e^(-2*pi*i*k*n/N), escrito como producto matricial
    return np.exp(-2j * np.pi * k * n / N) @ x


# ---------------------------------------------------------------------
# 2. FFT 2D: FFT sobre las filas y luego sobre las columnas
# ---------------------------------------------------------------------

# funcion que calcula la FFT 2D de una imagen (o de un lote de imagenes)
def fft2_recursiva(img):

    # FFT de cada fila (ultimo eje)
    por_filas = fft_recursiva(img)

    # se intercambian filas y columnas, se transforma, y se regresa a la forma original
    return fft_recursiva(por_filas.swapaxes(-1, -2)).swapaxes(-1, -2)


# funcion que calcula la FFT 2D inversa
def ifft2_recursiva(F):

    # IFFT de cada fila
    por_filas = ifft_recursiva(F)

    # IFFT de cada columna (intercambiando ejes igual que en la directa)
    return ifft_recursiva(por_filas.swapaxes(-1, -2)).swapaxes(-1, -2)


# funcion que lleva la frecuencia cero (DC) al centro del espectro (solo reordena)
def centrar_espectro(F):

    # fftshift solo reordena cuadrantes, no calcula ninguna transformada
    return np.fft.fftshift(F, axes=(-2, -1))


# funcion inversa de la anterior
def descentrar_espectro(F):

    # ifftshift deshace el reordenamiento
    return np.fft.ifftshift(F, axes=(-2, -1))


# ---------------------------------------------------------------------
# 3. IMAGENES: carga, ajuste a potencia de 2 e imagen sintetica de respaldo
# ---------------------------------------------------------------------

# funcion que lista las imagenes de una carpeta
def listar_imagenes(carpeta):
    
    # extensiones aceptadas
    extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # carpeta como objeto Path
    carpeta = Path(carpeta)

    # si la carpeta no existe se devuelve una lista vacia
    if not carpeta.exists():
        return []
    
    # archivos con extension valida, ordenados por nombre
    return sorted(p for p in carpeta.iterdir() if p.suffix.lower() in extensiones)


# funcion que carga una imagen, la ajusta a tam x tam y la normaliza a [0, 1]
def cargar_imagen(ruta, tam=256, gris=True):

    # apertura de la imagen
    im = Image.open(ruta)

    # conversion a escala de grises (L) o color (RGB)
    im = im.convert("L" if gris else "RGB")

    # el redimensionado a potencia de 2 es necesario para la FFT recursiva
    im = im.resize((tam, tam), Image.LANCZOS)

    # conversion a decimales entre 0 y 1
    return np.asarray(im, dtype=float) / 255.0


# funcion que genera una imagen de prueba con bordes, texturas y zonas suaves
def imagen_sintetica(tam=256):

    # malla de coordenadas normalizadas entre 0 y 1
    y, x = np.mgrid[0:tam, 0:tam] / tam

    # fondo con gradiente suave
    img = 0.3 + 0.2 * x + 0.1 * y

    # circulo brillante con borde nitido
    img[(x - 0.3) ** 2 + (y - 0.35) ** 2 < 0.04] = 0.9

    # rectangulo oscuro con borde nitido
    img[(0.55 < x) & (x < 0.9) & (0.15 < y) & (y < 0.45)] = 0.1

    # franjas verticales (textura de alta frecuencia)
    franjas = (0.1 < x) & (x < 0.5) & (0.6 < y) & (y < 0.9)
    img[franjas] = 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 24 * x[franjas]))

    # patron de tablero de ajedrez
    tablero = (0.6 < x) & (x < 0.9) & (0.6 < y) & (y < 0.9)
    img[tablero] = 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 12 * x[tablero]) * np.sin(2 * np.pi * 12 * y[tablero]))
    
    # se asegura el rango [0, 1]
    return np.clip(img, 0, 1)


# funcion que aplica una funcion de filtrado a cada canal si la imagen es de color
def aplicar_por_canal(funcion, img, *args, **kwargs):
    # imagen en gris: se aplica directo
    if img.ndim == 2:
        return funcion(img, *args, **kwargs)
    # imagen a color: se filtra cada canal por separado y se vuelven a apilar
    return np.stack([funcion(img[..., c], *args, **kwargs) for c in range(img.shape[-1])], axis=-1)


# ---------------------------------------------------------------------
# 4. FILTROS EN FRECUENCIA
# Convencion de frecuencia de corte (declarar en el reporte):
#   radio normalizado r en [0, 1] medido desde el centro del espectro centrado.
#   r = 0 es la frecuencia cero (DC) y r = 1 es la esquina (frecuencia mas alta).
# ---------------------------------------------------------------------

# funcion que calcula el radio normalizado de cada punto del espectro centrado
def radio_normalizado(forma):
    # alto y ancho de la imagen
    H, W = forma
    # coordenadas verticales y horizontales respecto al centro (indice H//2, W//2)
    fy = (np.arange(H) - H // 2) / (H / 2)
    fx = (np.arange(W) - W // 2) / (W / 2)
    # distancia al centro; se divide entre raiz de 2 para que la esquina valga 1
    return np.sqrt(fx[None, :] ** 2 + fy[:, None] ** 2) / np.sqrt(2)


# funcion que construye la mascara del filtro pasa bajas
def mascara_pasabajas(forma, fc, tipo="ideal"):
    # radio normalizado de cada punto
    r = radio_normalizado(forma)
    # filtro ideal: deja pasar todo lo que esta dentro del radio fc
    if tipo == "ideal":
        return (r <= fc).astype(float)
    # filtro gaussiano: atenuacion suave (sin el efecto de anillos del ideal)
    if tipo == "gaussiano":
        return np.exp(-(r ** 2) / (2 * fc ** 2))
    # tipo no reconocido
    raise ValueError("tipo debe ser 'ideal' o 'gaussiano'")


# funcion que construye la mascara del filtro pasa altas (complemento del pasa bajas)
def mascara_pasaaltas(forma, fc, tipo="ideal"):
    # lo que no pasa en el pasa bajas, pasa en el pasa altas
    return 1.0 - mascara_pasabajas(forma, fc, tipo)


# funcion que aplica una mascara a una imagen en gris usando la FFT propia
def filtrar_con_mascara(img, mascara):
    # FFT 2D propia y centrado del espectro
    F = centrar_espectro(fft2_recursiva(img))
    # multiplicacion del espectro por la mascara
    F_filtrada = F * mascara
    # regreso al dominio espacial con la IFFT 2D propia
    resultado = ifft2_recursiva(descentrar_espectro(F_filtrada))
    # la parte imaginaria es solo error numerico; se conserva la real
    return resultado.real


# funcion que aplica un pasa bajas a una imagen (gris o color)
def filtro_pasabajas(img, fc, tipo="ideal"):
    # mascara segun el tamano de los dos primeros ejes
    mascara = mascara_pasabajas(img.shape[:2], fc, tipo)
    # filtrado canal por canal y recorte al rango valido [0, 1]
    return np.clip(aplicar_por_canal(filtrar_con_mascara, img, mascara), 0, 1)


# funcion que aplica un pasa altas a una imagen (gris o color)
def filtro_pasaaltas(img, fc, tipo="ideal"):
    # mascara segun el tamano de los dos primeros ejes
    mascara = mascara_pasaaltas(img.shape[:2], fc, tipo)
    # el resultado puede tener valores negativos (no tiene componente DC), no se recorta
    return aplicar_por_canal(filtrar_con_mascara, img, mascara)


# funcion que escala una imagen a [0, 1] solo para poder mostrarla
def normalizar_para_mostrar(img):
    # minimo y maximo de la imagen
    mn, mx = img.min(), img.max()
    # si es constante se devuelve ceros para evitar dividir entre 0
    return np.zeros_like(img) if mx == mn else (img - mn) / (mx - mn)


# funcion que desenfoca una imagen con un pasa bajas gaussiano (sirve para pruebas)
def desenfoque_gaussiano(img, fc=0.05):
    # un fc pequeno elimina casi todas las frecuencias altas
    return filtro_pasabajas(img, fc, tipo="gaussiano")


# ---------------------------------------------------------------------
# 5. BLUR DETECTION
# Idea: una imagen nitida tiene bordes, y los bordes son frecuencias altas.
# Si falta energia en altas frecuencias, la imagen esta desenfocada.
# ---------------------------------------------------------------------

# funcion que mide que fraccion de la energia espectral esta en altas frecuencias
def puntaje_blur_fft(img, fc_alta=0.1):
    # si la imagen es de color se pasa a gris con promedio de canales
    if img.ndim == 3:
        img = img.mean(axis=-1)
    # espectro centrado y potencia |F|^2
    potencia = np.abs(centrar_espectro(fft2_recursiva(img))) ** 2
    # radio normalizado de cada punto
    r = radio_normalizado(img.shape)
    # se ignora la componente DC (brillo promedio), que no aporta informacion de nitidez
    potencia[img.shape[0] // 2, img.shape[1] // 2] = 0
    # energia total sin DC
    total = potencia.sum()
    # fraccion de la energia que cae fuera del radio fc_alta
    return float(potencia[r > fc_alta].sum() / total) if total > 0 else 0.0


# funcion de comprobacion: varianza del Laplaciano (metodo clasico de nitidez)
def varianza_laplaciano(img):
    # si la imagen es de color se pasa a gris con promedio de canales
    if img.ndim == 3:
        img = img.mean(axis=-1)
    # Laplaciano discreto con 4 vecinos, calculado con desplazamientos de la matriz
    lap = (-4 * img[1:-1, 1:-1] + img[:-2, 1:-1] + img[2:, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:])
    # nitidez = varianza del Laplaciano
    return float(lap.var())


# funcion que decide si una imagen esta desenfocada comparando el puntaje con un umbral
# umbral=0.02 se obtuvo con imagenes de prueba (nitidas: 0.08 a 0.43; muy desenfocadas: < 0.01)
# el puntaje depende del contenido, por eso hay que recalibrarlo con las imagenes del proyecto
def detectar_desenfoque(img, umbral=0.02, fc_alta=0.1):
    # puntaje de energia en altas frecuencias
    puntaje = puntaje_blur_fft(img, fc_alta)
    # por debajo del umbral falta energia de bordes: desenfocada
    return {"puntaje_fft": puntaje, "laplaciano": varianza_laplaciano(img), "desenfocada": puntaje < umbral}
