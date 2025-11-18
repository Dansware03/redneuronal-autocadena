# Evolución y Estadísticas de Modelos de Detección de Cadenas

Este documento detalla el proceso iterativo de entrenamiento y mejora de los modelos de detección de objetos para la identificación de cadenas nacionales. Cada modelo representa una etapa en el ciclo de desarrollo, con cambios en el dataset, los parámetros de entrenamiento o la estrategia de modelado.

---

## Modelo 1: Línea de Base Inicial

Este fue el primer intento, utilizando un conjunto de datos muy pequeño para establecer una línea de base y validar el pipeline.

*   **Descripción:** Modelo inicial con datos limitados.
*   **Modelo Base:** YOLOv8-Medium (`yolov8m`)
*   **Estrategia de Clases:** 2 Clases (`intro`, `outro`)
*   **Tamaño de Imagen (`imgsz`):** 640x640 px
*   **Dataset:**
    *   **Entrenamiento:** ~108 imágenes
    *   **Validación:** ~96 imágenes
*   **Métricas de Rendimiento (Validación):**
    | Métrica (Promedio `all`) | Valor   |
    | :----------------------- | :------ |
    | **mAP50**                | `0.188` |
    | **mAP50-95**             | `0.076` |
    | **Precisión (P)**        | `0.299` |
    | **Recall (R)**           | `0.326` |
*   **Análisis:** El rendimiento fue **muy bajo**, como era de esperar. El modelo apenas aprendió a distinguir las clases. Sin embargo, sirvió para confirmar que la estructura del proyecto era correcta. **Conclusión: Se necesita un aumento masivo de datos.**

---

## Modelo 3: Aumento de Datos y Resolución

Este modelo representa el primer gran salto en calidad, abordando directamente las limitaciones del Modelo 1.

*   **Descripción:** Modelo entrenado con un dataset significativamente más grande y a una resolución mayor.
*   **Modelo Base:** YOLOv8-Medium (`yolov8m`)
*   **Estrategia de Clases:** 2 Clases (`intro`, `outro`)
*   **Tamaño de Imagen (`imgsz`):** `736x736` px (ajustado desde 720)
*   **Dataset:**
    *   **Entrenamiento:** 630 imágenes (420 positivas, 210 negativas)
    *   **Validación:** 89 imágenes (63 positivas, 26 negativas)
*   **Métricas de Rendimiento (Validación):**
    | Métrica (Promedio `all`) | Valor   |
    | :----------------------- | :------ |
    | **mAP50**                | `0.675` |
    | **mAP50-95**             | `0.445` |
    | **Precisión (P)**        | `0.579` |
    | **Recall (R)**           | `0.764` |
*   **Análisis:** Un **salto de calidad espectacular**. El `mAP50` se triplicó, demostrando el impacto de un buen dataset y una mayor resolución. El modelo desarrolló un perfil de "Vigilante Ansioso": muy bueno para encontrar casi todas las cadenas (`Recall` alto), pero propenso a generar falsas alarmas (`Precisión` moderada). Se observó que la confusión entre `intro` y `outro` era una fuente de error.

---

## Modelo 5: Estrategia de Clase Única

Este modelo probó la hipótesis de que simplificar el problema para el modelo mejoraría la fiabilidad, a costa de no diferenciar entre inicio y fin de la cadena.

*   **Descripción:** Modelo entrenado con las clases `intro` y `outro` fusionadas en una sola clase `cadena`.
*   **Modelo Base:** YOLOv8-Medium (`yolov8m`)
*   **Estrategia de Clases:** 1 Clase (`cadena`)
*   **Tamaño de Imagen (`imgsz`):** `640x640` px
*   **Dataset:**
    *   **Entrenamiento:** 846 imágenes (636 positivas, 210 negativas)
    *   **Validación:** 233 imágenes (106 positivas, 127 negativas)
*   **Métricas de Rendimiento (Validación):**
    | Métrica (Promedio `all`) | Valor   |
    | :----------------------- | :------ |
    | **mAP50**                | `0.648` |
    | **mAP50-95**             | `0.514` |
    | **Precisión (P)**        | `0.735` |
    | **Recall (R)**           | `0.631` |
*   **Análisis:** La hipótesis fue correcta. Al eliminar la ambigüedad, la **Precisión se disparó** a `0.735`, creando un modelo mucho más fiable y con menos falsos positivos. Sin embargo, el `Recall` y el `mAP50` general disminuyeron porque el modelo se volvió más "cauteloso" y se entrenó a una resolución menor. **Conclusión: La estrategia de clase única es superior para la fiabilidad.**

---

## Modelo 6: El Modelo Campeón (Síntesis Final)

Este modelo combina las lecciones aprendidas de todos los anteriores: el gran dataset del M3, la estrategia de clase única del M5 y la alta resolución de entrenamiento del M3.

*   **Descripción:** Modelo final que utiliza una clase unificada y entrena a alta resolución con el dataset más grande.
*   **Modelo Base:** YOLOv8-Medium (`yolov8m`)
*   **Estrategia de Clases:** 1 Clase (`cadena`)
*   **Tamaño de Imagen (`imgsz`):** `736x736` px
*   **Dataset:**
    *   **Entrenamiento:** 846 imágenes (636 positivas, 210 negativas)
    *   **Validación:** 233 imágenes (106 positivas, 127 negativas)
*   **Métricas de Rendimiento (Validación):**
    | Métrica (Promedio `all`) | Valor   |
    | :----------------------- | :------ |
    | **mAP50**                | `0.683` |
    | **mAP50-95**             | `0.538` |
    | **Precisión (P)**        | `0.690` |
    | **Recall (R)**           | `0.745` |
*   **Análisis:** Este modelo logró el **mejor equilibrio de todos**. Obtuvo el `mAP50` y `mAP50-95` más altos, indicando la mejor calidad general. Logró una precisión casi tan alta como la del M5, pero con un `Recall` significativamente mejor. Representa el "punto dulce" óptimo.

---

## Modelo 7: Refinamiento con Dataset Extendido

Este modelo es una evolución directa del Modelo 6, manteniendo la misma configuración pero entrenando con un conjunto de datos aún más grande y refinado.

*   **Descripción:** Modelo final entrenado con el dataset más completo para maximizar el rendimiento.
*   **Modelo Base:** YOLOv8-Medium (`yolov8m`)
*   **Estrategia de Clases:** 1 Clase (`cadena`)
*   **Tamaño de Imagen (`imgsz`):** `736x736` px
*   **Dataset:**
    *   **Entrenamiento:** 984 imágenes (774 positivas, 210 negativas)
    *   **Validación:** 337 imágenes (210 positivas, 127 negativas)
*   **Métricas de Rendimiento (Validación):**
    | Métrica (Promedio `all`) | Valor   |
    | :----------------------- | :------ |
    | **mAP50**                | `0.782` |
    | **mAP50-95**             | `0.644` |
    | **Precisión (P)**        | `0.785` |
    | **Recall (R)**           | `0.730` |
*   **Análisis:** Este modelo representa el pináculo del rendimiento. El aumento de datos resultó en un **salto masivo en todas las métricas de calidad**. El `mAP50` y `mAP50-95` alcanzaron niveles de alta producción. Notablemente, la **Precisión** aumentó drásticamente a `0.785`, haciendo que el modelo sea extremadamente fiable y genere muy pocos falsos positivos, mientras mantiene un `Recall` muy alto y robusto.

---

## Modelo 8: Experimentación con Arquitectura v11-Medium

Este modelo explora el rendimiento de una arquitectura más reciente (`YOLOv11m`), utilizando el mismo dataset y configuración de entrenamiento que el Modelo 7 para una comparación directa.

*   **Descripción:** Experimento con una arquitectura de modelo diferente para evaluar mejoras potenciales.
*   **Modelo Base:** YOLOv11-Medium (`yolo11m`)
*   **Estrategia de Clases:** 1 Clase (`cadena`)
*   **Tamaño de Imagen (`imgsz`):** `736x736` px
*   **Dataset:**
    *   **Entrenamiento:** 984 imágenes (774 positivas, 210 negativas)
    *   **Validación:** 337 imágenes (210 positivas, 127 negativas)
*   **Métricas de Rendimiento (Validación):**
    | Métrica (Promedio `all`) | Valor   |
    | :----------------------- | :------ |
    | **mAP50**                | `0.786` |
    | **mAP50-95**             | `0.640` |
    | **Precisión (P)**        | `0.798` |
    | **Recall (R)**           | `0.720` |
*   **Análisis:** El cambio a la arquitectura `YOLOv11m` produjo resultados marginalmente superiores en `mAP50` y `Precisión`, a costa de una ligera caída en `Recall` y `mAP50-95`. Esto sugiere que la nueva arquitectura es ligeramente más precisa pero potencialmente un poco menos robusta en la localización exacta de los objetos. Los modelos son **prácticamente equivalentes en rendimiento**, lo que indica que para este dataset específico, la mejora principal sigue proviniendo de la calidad de los datos más que de la arquitectura del modelo.

---

## Modelo 9: Optimización de Velocidad (v11-Nano)

Este modelo investiga si una arquitectura más pequeña y rápida (`YOLOv11n`) puede alcanzar un rendimiento comparable a los modelos medianos, entrenando durante más épocas para compensar su menor capacidad.

*   **Descripción:** Entrenamiento con un modelo 'Nano' durante 100 épocas para buscar un equilibrio óptimo entre velocidad y precisión.
*   **Modelo Base:** YOLOv11-Nano (`yolo11n`)
*   **Estrategia de Clases:** 1 Clase (`cadena`)
*   **Tamaño de Imagen (`imgsz`):** `736x736` px
*   **Dataset:**
    *   **Entrenamiento:** 984 imágenes (774 positivas, 210 negativas)
    *   **Validación:** 337 imágenes (210 positivas, 127 negativas)
*   **Métricas de Rendimiento (Validación):**
    | Métrica (Promedio `all`) | Valor   |
    | :----------------------- | :------ |
    | **mAP50**                | `0.770` |
    | **mAP50-95**             | `0.634` |
    | **Precisión (P)**        | `0.811` |
    | **Recall (R)**           | `0.708` |
*   **Análisis:** Un resultado **sorprendentemente bueno**. El modelo `YOLOv11n` logró un `mAP50` casi idéntico al de sus contrapartes medianas, pero con una **Precisión notablemente más alta (`0.811`)**. Esto lo convierte en el modelo más fiable de todos en términos de falsos positivos. Su `Recall` es ligeramente más bajo, pero la velocidad de inferencia (`~3.4ms`) es **más del doble de rápida** que la de los modelos medianos.

---

## Tabla Comparativa Final

| Métrica (Promedio `all`) | M1 (Base) | M3 (Bueno) | M5 (Preciso) | M6 (YOLOv8m) | M7 (YOLOv8m) | M8 (YOLOv11m) | **M9 (YOLOv11n)** |
| :----------------------- | :-------- | :--------- | :----------- | :------------- | :------------- | :-------------- | :------------------ |
| **mAP50** (Rendimiento)    | 0.188     | 0.675      | 0.648        | 0.683          | 0.782          | **`0.786`**     | 0.770               |
| **mAP50-95** (Robustez)  | 0.076     | 0.445      | 0.514        | 0.538          | **`0.644`**    | 0.640           | 0.634               |
| **Precisión (P)**            | 0.299     | 0.579      | 0.735        | 0.690          | 0.785          | 0.798           | **`0.811`**         |
| **Recall (R)**               | 0.326     | **`0.764`**| 0.631        | 0.745          | 0.730          | 0.720           | 0.708               |
| **Inferencia (ms)**        | -         | -          | ~7.5         | ~8.9           | ~7.5           | ~6.9            | **`~3.4`**          |


### **Conclusión General**

El viaje desde el Modelo 1 hasta el **Modelo 9** demuestra un ciclo de desarrollo de IA exitoso. Las mejoras no vinieron de cambiar el modelo en sí, sino de un enfoque metódico en la ingeniería de datos y la estrategia de modelado:
1.  **Los datos son el factor más importante:** El salto de M1 a M3 y de M6 a M7 lo prueba de manera concluyente.
2.  **Simplificar el problema para el modelo funciona:** La unificación de clases fue clave para mejorar drásticamente la precisión.
3.  **La resolución de entrenamiento importa:** Entrenar con imágenes más grandes permitió al modelo capturar más detalles.
4.  **El tamaño del modelo es un trade-off:** El **Modelo 9 (`YOLOv11n`)** emerge como el **campeón inesperado**. Aunque su `mAP50` es ligeramente inferior al del M8, ofrece la **mejor Precisión** de todos los modelos y es **dos veces más rápido**. Para una aplicación en tiempo real donde la fiabilidad (pocos falsos positivos) y la eficiencia son cruciales, el Modelo 9 es la elección técnica superior.

El **Modelo 9 (`yolo11n`)** es el resultado de una optimización completa, demostrando que un modelo más pequeño y rápido, con suficiente tiempo de entrenamiento, puede superar a sus contrapartes más grandes en las métricas que más importan para la producción.