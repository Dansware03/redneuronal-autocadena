# Análisis de Causa Raíz: Discrepancia en el Pre-procesamiento de Imágenes

## 1. Resumen Ejecutivo

Se ha identificado un error crítico que causaba que la segunda capa de verificación (Red Neuronal) validara incorrectamente frames de programación normal como si fueran de la cadena nacional, resultando en una tasa de falsos positivos inaceptable (`10/10 frames coincidentes`).

El análisis concluye que el problema **no es un error en el modelo de la red neuronal ni en la lógica de verificación**, sino una **discrepancia fundamental en el pre-procesamiento de las imágenes** entre el entorno de la aplicación en producción y el script utilizado para el entrenamiento y la validación externa.

## 2. El Problema: Dos Pipelines de Transformación Diferentes

La clave del problema reside en cómo se preparan las imágenes antes de ser analizadas por el modelo MobileNetV3.

### 2.1. Pipeline del Script de Entrenamiento y Validación (Correcto)

El script `embedding_extractor.py`, utilizado para generar el índice FAISS y para las pruebas externas, utiliza el siguiente pipeline de transformación de `torchvision`:

```python
# embedding_extractor.py
self.transform = T.Compose([
    T.ToPILImage(),
    T.Resize((image_size, image_size)), # <- Redimensiona a 224x224
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])
```

-   **Comportamiento:** La imagen completa se redimensiona a 224x224 píxeles. El modelo "ve" la imagen entera, con todo su contexto. **Este es el método correcto y consistente con el que se entrenó el sistema.**

### 2.2. Pipeline de la Aplicación en Producción (Incorrecto)

El archivo `app/vision/verifier.py`, que se ejecuta dentro de la aplicación, utilizaba un pipeline diferente, estándar para modelos pre-entrenados en ImageNet, pero incorrecto para este caso de uso:

```python
# app/vision/verifier.py (versión anterior)
self.transform = transforms.Compose([
    transforms.Resize(256),             # <- Redimensiona a 256x256
    transforms.CenterCrop(224),         # <- RECORTA el centro a 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[...], std=[...]),
])
```

-   **Comportamiento:** La imagen se redimensiona a un tamaño mayor (256x256) y luego **se recorta el centro de la imagen**, descartando los bordes. El modelo solo "ve" una porción central de la imagen original.

## 3. Consecuencias de la Discrepancia

Esta diferencia en el pre-procesamiento es la causa directa del error:

1.  **Pérdida de Contexto:** Al recortar el centro, se pueden eliminar elementos visuales clave que diferencian la programación normal de la cadena (logos, gráficos, etc.), que a menudo se encuentran en los bordes.
2.  **"Falso Foco":** Si el centro de una imagen de programación normal contiene, por casualidad, elementos visuales que se parecen a los del centro de una intro (ej. un paisaje, un color similar), el modelo se "confunde" y genera un embedding (firma visual) erróneo.
3.  **Inconsistencia Total:** El modelo en producción estaba comparando "firmas de imágenes recortadas" con una base de datos de "firmas de imágenes completas". Esta inconsistencia invalida por completo el proceso de verificación y es la razón por la cual el script de prueba externo funcionaba correctamente (usando el pipeline correcto) mientras que la aplicación fallaba.

## 4. Solución Implementada y Optimizaciones Adicionales

La solución ha consistido en alinear y optimizar el pipeline de pre-procesamiento y verificación en `app/vision/verifier.py`.

### 4.1. Corrección del Pipeline de Pre-procesamiento

Se ha modificado el pipeline de `torchvision.transforms` para que sea idéntico al utilizado durante el entrenamiento, eliminando el `Resize(256)` y el `CenterCrop(224)` y sustituyéndolos por un `Resize((224, 224))` directo. Esto asegura que la imagen completa, sin recortes, sea analizada.

### 4.2. Optimización por Procesamiento en Lote (Batch Processing)

Se ha refactorizado la lógica de verificación para que, en lugar de procesar los 15 frames uno por uno, se apilen todos en un único tensor ("batch"). Este lote se envía a la red neuronal en una sola operación, lo que reduce drásticamente la sobrecarga y acelera significativamente el proceso de verificación al aprovechar la paralelización de la CPU/GPU.

### 4.3. Implementación de Normalización L2

Se ha añadido un paso de **Normalización L2** a los embeddings generados por el modelo. Esta es una mejora crítica que asegura que todos los vectores de características tengan una longitud unitaria. Como resultado, la búsqueda por producto interno (`IndexFlatIP`) en FAISS se convierte matemáticamente en una búsqueda por **similitud de coseno**, que es mucho más robusta y hace que el umbral de similitud (`0.85`) sea consistente y fiable.

## 5. ¡ACCIÓN REQUERIDA! Necesidad de Reconstruir el Índice FAISS

**ADVERTENCIA CRÍTICA:** Debido a la implementación de la **Normalización L2**, el formato de los embeddings que genera la aplicación ha cambiado. Los embeddings antiguos en tu índice FAISS no están normalizados.

**Es absolutamente necesario que reconstruyas tu índice FAISS** utilizando el script de entrenamiento actualizado con esta misma lógica de normalización L2. Si no lo haces, el sistema no funcionará, ya que estará comparando vectores normalizados (de la aplicación) con vectores no normalizados (del índice), lo que resultará en que ninguna imagen coincida.