# Análisis Técnico del Módulo `FingerprintVerifier`

## 1. Introducción

El módulo `FingerprintVerifier`, ubicado en `app/vision/verifier.py`, constituye la segunda capa de verificación del sistema de detección de cadenas. Su propósito principal es confirmar, con alta precisión, si una secuencia de fotogramas de vídeo corresponde a una cortinilla de "intro" (inicio de cadena) o de "outro" (fin de cadena) previamente identificada por un modelo de inteligencia artificial de primera capa.

Para lograr esto, utiliza una técnica de huella digital perceptual (p-hash), que es robusta frente a pequeñas variaciones en los fotogramas, como ligeros cambios de color, compresión o escalado.

## 2. Inicialización y Carga de Huellas

### `__init__(self, fingerprint_data: dict, ai_config: dict)`

El constructor de la clase `FingerprintVerifier` recibe dos argumentos:

1.  `fingerprint_data`: Un diccionario que contiene las huellas digitales (hashes) precalculadas de las cortinillas conocidas. Este diccionario debe tener dos claves:
    *   `"intro_hashes"`: Una lista de strings, donde cada string es la representación hexadecimal de un p-hash de un fotograma de la cortinilla de "intro".
    *   `"outro_hashes"`: Una lista similar para la cortinilla de "outro".
2.  `ai_config`: Un diccionario con parámetros de configuración que ajustan la sensibilidad del algoritmo de verificación.

### `_process_fingerprints(self, data: dict)`

Este método interno es llamado por el constructor para procesar los datos de `fingerprint_data`. Su función es:

1.  **Validar**: Comprueba si se han proporcionado datos de huellas.
2.  **Convertir**: Itera sobre las listas de hashes en formato hexadecimal y las convierte en objetos `ImageHash` utilizando la función `imagehash.hex_to_hash()`. Estos objetos son más eficientes para realizar comparaciones.
3.  **Almacenar**: Guarda las listas de objetos `ImageHash` en los atributos `self.db_intro_hashes` y `self.db_outro_hashes`.

Este pre-procesamiento es crucial para que las comparaciones posteriores se realicen de manera rápida y eficiente.

## 3. Proceso de Verificación

### `verify(self, frames: list, is_cadena_currently_active: bool) -> str`

Este es el método público principal del verificador. Orquesta todo el proceso de análisis de una nueva secuencia de vídeo.

#### **Paso 1: Recepción y Conversión de Frames**

El método recibe una lista de `frames` (imágenes en formato de array NumPy, provenientes de OpenCV) y `is_cadena_currently_active`, un booleano que indica si el sistema ya se encuentra en estado de "cadena activa".

Lo primero que hace es convertir cada fotograma (frame) en un p-hash. Este proceso sigue los siguientes sub-pasos para cada frame:

1.  **Conversión de Color**: El formato de color se cambia de BGR (usado por OpenCV) a RGB.
    ```python
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ```
2.  **Creación de Imagen PIL**: El array NumPy se convierte en un objeto de imagen de la librería `PIL` (Pillow).
    ```python
    Image.fromarray(...)
    ```
3.  **Conversión a Escala de Grises**: La imagen se convierte a escala de grises (`'L'`) para simplificarla y hacer el hash insensible al color.
    ```python
    .convert('L')
    ```
4.  **Cálculo del p-hash**: Finalmente, se calcula el p-hash de la imagen en escala de grises.
    ```python
    imagehash.phash(img)
    ```

El resultado de este paso es una lista de objetos `ImageHash` llamada `captured_hashes`, que representa la "huella digital" de la secuencia de vídeo recibida.

#### **Paso 2: Búsqueda Dirigida por Estado**

El verificador optimiza la búsqueda basándose en el estado actual de la aplicación (`is_cadena_currently_active`):

*   **Si la cadena está activa (`True`)**: El sistema solo puede estar esperando una cortinilla de "outro". Por lo tanto, la búsqueda se realiza exclusivamente contra la base de datos `self.db_outro_hashes`.
*   **Si la cadena no está activa (`False`)**: El sistema solo busca una cortinilla de "intro", comparando los hashes capturados contra `self.db_intro_hashes`.

Esta lógica evita búsquedas innecesarias y previene falsos positivos (ej. detectar un "intro" cuando se esperaba un "outro").

#### **Paso 3: Algoritmo de Coincidencia Agrupada (`_find_clustered_match`)**

La decisión final de si hay una coincidencia o no se delega al método `_find_clustered_match`. Este método determina si los `captured_hashes` coinciden de forma significativa con la base de datos de hashes seleccionada (intro u outro).

El resultado final del método `verify` es un string: `"intro"`, `"outro"` o `"none"`.

## 4. Algoritmo de Coincidencia: `_find_clustered_match`

Este método es el núcleo del verificador y determina si una secuencia de hashes capturados es una coincidencia válida. No basta con encontrar hashes similares; estos deben aparecer de forma "agrupada" o "concentrada" en la base de datos, reflejando la naturaleza secuencial de un vídeo.

#### **Paso 1: Búsqueda de Coincidencias Individuales**

El algoritmo itera a través de cada hash capturado (`h_captured`) y lo compara con todos los hashes de la base de datos (`h_db`).

La comparación se realiza calculando la **distancia de Hamming** entre los dos hashes. Esta distancia representa cuántos bits son diferentes entre un hash y otro.
```python
if (h_captured - h_db) <= hash_tolerance:
```
*   `hash_tolerance`: Es un umbral configurable (ej. 6). Si la distancia de Hamming es menor o igual a este umbral, se considera una coincidencia. Esto da robustez al sistema, permitiendo pequeñas variaciones en los fotogramas.

Cuando se encuentra una coincidencia para un hash capturado, se guarda el **índice** de su correspondiente en la base de datos (`i`) en una lista llamada `match_indices`, y la búsqueda para ese hash capturado se detiene (`break`).

#### **Paso 2: Verificación de Criterios de Validez**

Una vez recopilados todos los índices de las coincidencias, se aplican dos filtros cruciales:

1.  **Suficientes Coincidencias (`MIN_MATCH_COUNT`)**:
    ```python
    if len(match_indices) < min_match_count:
        return False
    ```
    Se comprueba si el número total de coincidencias encontradas es superior a un mínimo requerido (`min_match_count`, ej. 5). Esto evita que unas pocas coincidencias aisladas y posiblemente aleatorias disparen una detección positiva.

2.  **Coincidencias Agrupadas (`MAX_MATCH_SKEW`)**:
    ```python
    match_span = max(match_indices) - min(match_indices)
    if match_span >= max_match_skew:
        return False
    ```
    Este es el criterio más importante. Se calcula la "dispersión" (`match_span`) de las coincidencias, que es la diferencia entre el índice más alto y el más bajo encontrados en la base de datos.
    *   Si esta dispersión es mayor o igual a `max_match_skew` (ej. 20), significa que las coincidencias están muy "desordenadas" o esparcidas a lo largo de la base de datos de huellas. Esto no es representativo de una secuencia de vídeo coherente, por lo que se descarta.
    *   Si la dispersión es pequeña, implica que los fotogramas capturados coinciden con una subsecuencia contigua y ordenada de la base de datos, lo cual es una fuerte señal de una verificación positiva.

Si ambos criterios se cumplen, el método devuelve `True`, confirmando la detección.

## 5. Optimización con Cython

El código muestra un patrón de diseño para la optimización de rendimiento. Intenta importar una versión compilada de la función `_find_clustered_match` desde un módulo Cython (`verifier_cython`).

*   **Si la importación tiene éxito (`CYTHON_AVAILABLE = True`)**: La lógica de búsqueda, que es computacionalmente intensiva debido a los bucles anidados, se delega a la función de Cython, que se ejecuta a velocidades cercanas al C nativo.
*   **Si la importación falla**: El sistema recurre de forma transparente a la implementación en Python puro descrita anteriormente. Esto asegura que el código funcione siempre, incluso si no ha sido compilado.
