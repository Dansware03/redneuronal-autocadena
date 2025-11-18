# Análisis de la Red Neuronal

## 2. Funcionamiento de la 2ª Capa Actual (MobileNetV3 + FAISS)

Es fundamental detallar el funcionamiento del sistema de verificación actual, que ha demostrado ser robusto contra los falsos positivos.

### 2.1. ¿Qué es y qué hace la Red Neuronal?

1.  **Generación de "Embeddings" (Firmas Visuales):**
    - El sistema utiliza el modelo de red neuronal **MobileNetV3**, pre-entrenado en un vasto conjunto de imágenes (ImageNet).
    - La función de este modelo no es decir "esto es una cadena" o "esto no es una cadena". Su única tarea es actuar como un **extractor de características universal**.
    - Cuando recibe un frame de vídeo, lo procesa y genera un vector de 1024 números llamado **"embedding"**. Este embedding es una firma numérica que representa la esencia visual del frame: sus formas, colores, texturas y la relación entre ellos.

2.  **Búsqueda por Similitud (FAISS):**
    - Durante el entrenamiento, se generaron los embeddings de más de 7,000 imágenes confirmadas de la intro/outro de la cadena nacional.
    - Estos embeddings se almacenaron en un índice especial llamado **FAISS (Facebook AI Similarity Search)**. FAISS es una base de datos optimizada para encontrar vectores similares a una velocidad extremadamente alta.
    - Cuando el sistema está en funcionamiento, toma el embedding del frame actual y le pregunta a FAISS: "¿Cuáles de los embeddings de cadena que tienes guardados se parecen más a este?".

### 2.2. Lógica de Decisión para Confirmar una Cadena

El sistema no toma una decisión basándose en un solo frame, sino en una secuencia para asegurar robustez.

1.  **Recolección de Frames:** Cuando la primera capa (YOLO) detecta una posible cadena, el sistema recolecta los siguientes **20 frames** del vídeo.
2.  **Verificación Individual:** Cada uno de estos 20 frames se convierte en un embedding y se compara con el índice FAISS.
3.  **Criterio de Aceptación:** Para que la cadena sea confirmada, se deben cumplir dos condiciones:
    - **Umbral de Similitud (`similarity_threshold`):** La similitud entre el embedding del frame y el de la base de datos debe ser de al menos **0.85**. Una similitud de 1.0 sería una coincidencia perfecta.
    - **Mínimo de Coincidencias (`min_matches`):** De los 20 frames analizados, al menos **8** deben superar este umbral de similitud.

En resumen, la segunda capa confirma una cadena si **al menos 8 de 20 frames son visualmente similares en un 85% o más** a alguna de las imágenes de referencia de la cadena.
