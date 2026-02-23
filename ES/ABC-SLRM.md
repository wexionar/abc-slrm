# ABC-SLRM: Modelo de Regresión Lineal Segmentada

**Sobre el Marco de Referencia ABC, sus Fundamentos y su Relación con SLRM**

**SLRM Team:**<br>
Alex · Gemini · ChatGPT<br>
Claude · Grok · Meta AI<br>
DeepSeek · GLM · Anna

**Versión:** 2.0<br>
**Licencia:** MIT<br>
**Fecha:** 2026-02

---

## PARTE 1

### 1. MARCO ABC

El Marco de Referencia ABC define tres fases universales que todo sistema de modelado de datos debe atravesar.

### 1.1 Fase A: El Dataset Base

- **Definición:** Es una colección de N registros en un espacio D-dimensional, donde cada registro contiene variables independientes: X = [X_1, X_2, ... , X_D] y una variable dependiente: Y. Se asume la relación funcional Y = f(X).

- **Ideal:** Lograr un conjunto de datos cuasi-infinito y geométricamente cuasi-continuo. 

- **Referencia:** dataset.csv

### 1.2 Fase B: Los Motores

Las herramientas que transforman y consultan los datos. Se dividen básicamente en tres tipos de motores:

#### B.1 - Motores Core

- **Definición:** Infieren sobre el Dataset Base.

- **Ideal:**  Inferir lo más rápido posible sobre Dataset Base siendo lo más preciso posible.

- **Referencia:** core.exe

#### B.2 - Motores Origin

- **Definición:** Optimizan/Comprimen el Dataset Base en Dataset Optimizado.

- **Ideal:** Optimizar/Comprimir lo máximo posible el Dataset Base perdiendo la mínima precisión posible.

- **Referencia:** origin.py

#### B.3 - Motores Resolution

- **Definición:** Infieren sobre el Dataset Optimizado.

- **Ideal:** Inferir lo más rápido posible sobre Dataset Optimizado siendo lo más preciso posible.

- **Referencia:** resolution.exe

### 1.3 Fase C: El Dataset Optimizado

- **Definición:** Es un Dataset Optimizado/Comprimido del Dataset Base, con igual o diferente estructura lógica-matemática.

- **Ideal:** Poder representar todos los datos del Dataset Base con la estructura lógica-matemática más sencilla posible.

- **Referencia:** dataset.npy

---

## PARTE 2

### 2. MODELOS ABC

Existen distintos modelos que siguen la estructura o anatomía del Marco de Referencia ABC.

### 2.1 El Modelo MML

- **Fase A:**<br>
└─ dataset.csv

- **Fase B:**<br>
├─ La Caja Negra<br>
└─ Etcétera

- **Fase C:**<br>
└─ dataset.h5   

### 2.2 El Modelo SLRM

- **Fase A:**<br>
└─ dataset.csv

- **Fase B.1:**<br>
├─ Atom Core<br>
├─ Logos Core<br>
├─ Lumin Core<br>
└─ Nexus Core

- **Fase B.2:**<br>
├─ Atom Origin<br>
├─ Logos Origin<br>
├─ Lumin Origin<br>
└─ Nexus Origin

- **Fase B.3:**<br>
├─ Atom Resolution<br>
├─ Logos Resolution<br>
├─ Lumin Resolution<br>
└─ Nexus Resolution

- **Fase C:**<br>
└─ dataset.npy

---

## PARTE 3

### 3. MOTORES SLRM

Los motores SLRM se organizan según el **soporte de puntos** necesario para realizar una inferencia. Esta jerarquía representa una progresión de complejidad creciente: desde la simplicidad de copiar el vecino más cercano (Atom), hasta la complejidad exponencial de requerir 2^D puntos para construir un politopo completo (Nexus).

### 3.1 Los Tres Tipos de Motores

- **B.1 - Motores Core**<br>
  ├─ Atom Core (soporte 1 punto)<br>
  ├─ Logos Core (soporte 2 puntos)<br>
  ├─ Lumin Core (soporte D+1 puntos)<br>
  └─ Nexus Core (soporte 2^D puntos)

- **B.2 - Motores Origin**<br>
  ├─ Atom Origin (soporte 1 punto)<br>
  ├─ Logos Origin (soporte 2 puntos)<br>
  ├─ Lumin Origin (soporte D+1 puntos)<br>
  └─ Nexus Origin (soporte 2^D puntos)
  
- **B.3 - Motores Resolution**<br>
  ├─ Atom Resolution (soporte 1 punto)<br>
  ├─ Logos Resolution (soporte 2 puntos)<br>
  ├─ Lumin Resolution (soporte D+1 puntos)<br>
  └─ Nexus Resolution (soporte 2^D puntos)

### 3.2 La Metáfora de las Hormigas

- **Los Motores Core son Hormigas Exploradoras:** descubren cómo inferir, identifican qué estructura necesitan, definen qué debe guardarse.

- **Los Motores Origin son Hormigas Constructoras:** siguen el camino marcado por Core, construyen el Dataset Optimizado.

- **Los Motores Resolution son Hormigas Obreras:** usan la estructura construida para inferir eficientemente.

### 3.3 La Analogía de la Arquitectura Fusion

- **Fusion** es una arquitectura que combina **dos motores en un contenedor.**

- Como un archivo .tar en Linux, **Fusion empaqueta dos motores** que trabajan en conjunto:

  - Atom Fusion = AtomOrigin + AtomResolution<br>
  - Logos Fusion = LogosOrigin + LogosResolution<br>
  - Lumin Fusion = LuminOrigin + LuminResolution<br>
  - Nexus Fusion = NexusOrigin + NexusResolution

---

### 3.4 Cuadro Resumen de Motores

| Tipos de Motores | Motores Core | Motores Origin | Motores Resolution |
|:-----------------|:------------:|:--------------:|:-----------------:|
| **Soporte 1 Punto** | Atom Core | Atom Origin | Atom Resolution |
| **Soporte 2 Puntos** | Logos Core | Logos Origin | Logos Resolution |
| **Soporte D+1 Puntos** | Lumin Core | Lumin Origin | Lumin Resolution |
| **Soporte 2^D Puntos** | Nexus Core | Nexus Origin | Nexus Resolution |
| **Contenedor Core** | X | - | - |
| **Contenedor Fusion** | - | X | X |

- **Atom** - El Especialista Buscador<br>
- **Logos** - El Especialista Unidimensional<br>
- **Lumin** - El Estándar Multidimensional<br>
- **Nexus** - El Especialista en Grids Densos

---

### 3.5 Tabla Comparativa de Motores Core

| Motor | Soporte | Estructura | Complejidad | Velocidad Relativa | Uso Ideal |
|-------|---------|------------|-------------|-------------------|-----------|
| **Atom** | 1 punto | Punto | O(log N) | **1.0× (más rápido)** ⚡ | Datasets extremadamente densos (N >> 10^6) |
| **Logos** | 2 puntos | Segmento | O(log N) | **1.4× más lento** | Series temporales 1D |
| **Lumin** | D+1 puntos | Simplex | O(log N + D²) | **3.0× más lento** | Datasets nD estándar |
| **Nexus** | 2^D puntos | Politopo | O(log N + D log D) | **4-5× más lento** | Grids estructurados (hasta ~15D) |

**Nota sobre Nexus:** Requiere un grid completo con 2^D puntos. En dimensiones altas, esto se vuelve exponencialmente inviable:
- 10D: 1,024 puntos (viable)
- 15D: 32,768 puntos (desafiante)
- 20D: 1,048,576 puntos (muy difícil)
- 50D: Prácticamente imposible (>10^15 puntos)

---

### 3.6 Progresión de Complejidad

La jerarquía Atom → Logos → Lumin → Nexus no solo representa incremento en puntos soporte, sino también en complejidad computacional:

| Motor | Puntos Soporte | Complejidad Búsqueda | Complejidad Cálculo | Complejidad Estructural | Velocidad Relativa |
|-------|----------------|---------------------|-------------------|------------------------|-------------------|
| **Atom** | 1 | O(log N) | O(1) - copia directa | **Mínima** | **1.0× (más rápido)** ⚡ |
| **Logos** | 2 | O(log N) | O(1) - interpolación 1D | **Baja** | **1.4× más lento** |
| **Lumin** | D+1 | O(log N) | O(D²) - sistema lineal | **Moderada** | **3.0× más lento** |
| **Nexus** | 2^D | O(log N) | O(D log D) - partición Kuhn | **Exponencial** | **4-5× más lento** |

---

## PARTE 4

### 4. CALIDAD SLRM

#### 4.1 EPSILON (ε)
En el modelo SLRM, el parámetro Epsilon (ε) se define estrictamente en función de la variable dependiente (Y). SLRM utiliza ε como un umbral de tolerancia sobre la respuesta (Y). 

**Nota de calidad:** Este parámetro determina la sensibilidad del modelo ante variaciones en la función f(X) y actúa como el criterio principal para la simplificación y linealización de los segmentos.

#### 4.2 INTERPOLACIÓN
Proceso de inferencia donde el punto de consulta se encuentra dentro de los límites definidos por la nube de datos conocidos.

**Nota de Práctica:** En SLRM la calidad de la inferencia depende de la calidad de datos del sector local a inferir, por lo tanto, se establece como buena práctica verificar la salud de dicho sector antes de emitir un resultado de Y.

#### 4.3 EXTRAPOLACIÓN
Proceso de inferencia donde el punto de consulta se encuentra fuera de los límites conocidos del Dataset para cada coordenada o eje X.

**Nota de Honestidad:** En SLRM, la extrapolación se asume como una proyección teórica. Al no existir un soporte físico de datos que respalde la consulta, el resultado de Y es una extensión de la tendencia local y debe ser tratado con cautela, reconociendo que la conjetura de linealidad no tiene verificación posible en este vacío.

---

## PARTE 5

### 5. GARANTÍAS Y FILOSOFÍA

#### 5.1 Garantías Matemáticas

SLRM proporciona **dos condiciones no negociables**:

1. **Condición 1 (Precisión de Entrenamiento):** Cualquier punto retenido en el modelo comprimido debe ser inferido con diferencia cero (o dentro de épsilon) 
2. **Condición 2 (Precisión de Puntos Descartados):** Cualquier punto descartado durante la compresión debe ser inferido dentro de épsilon, independientemente del orden de entrada.

Estas condiciones aseguran que la compresión del modelo no sacrifique la precisión.

#### 5.2 Filosofía de Diseño

1. **Eficiencia de Hardware:** Diseñado para CPUs y microcontroladores. La inferencia elimina la dependencia de la GPU. El entrenamiento es eficiente en CPU, aunque la aceleración por GPU es posible para la ingestión a gran escala.

2. **Transparencia Total:** Cada inferencia es rastreable a la ley lineal de un sector específico. Sin capas ocultas, sin transformaciones opacas—solo geometría local.

3. **El Simplex como un Átomo:** Así como la derivada linealiza una curva en un punto, el Simplex es la unidad mínima que garantiza una ley lineal pura en n-dimensiones sin introducir curvaturas artificiales.

4. **Lógica Determinista:** Sin descenso de gradiente, sin optimización estocástica, sin inicialización aleatoria. Mismo conjunto de datos + mismos parámetros = mismo modelo, siempre.

5. **Propósito:** Democratizar la alta precisión eliminando el costo computacional masivo del descenso de gradiente, devolviendo el modelado de datos al campo de la lógica determinista.

---

## PARTE 6

### 6. CONCLUSIÓN

SLRM representa un retorno a los **principios fundamentales geométricos** en el modelado de datos. Al reemplazar el descenso de gradiente con particionamiento determinista, logramos:

- **Transparencia:** Cada predicción es rastreable
- **Eficiencia:** Se ejecuta en CPUs y microcontroladores
- **Garantías:** Error limitado por épsilon, sin alucinaciones
- **Interpretabilidad:** Leyes lineales con significado físico

Esto no es un reemplazo para todas las redes neuronales, sino una **alternativa rigurosa** para aplicaciones donde la transparencia, la eficiencia y el determinismo importan más que exprimir el último 0.1% de precisión.

**La caja de cristal está abierta.**

---

## PARTE 7

### 7. BIBLIOGRAFÍA

1. **Kuhn, H.W. (1960).** "Some combinatorial lemmas in topology". *IBM Journal of Research and Development*.

2. **Munkres, J. (1984).** "Elements of Algebraic Topology". *Addison-Wesley*.

3. **Grünbaum, B. (2003).** "Convex Polytopes". *Springer Graduate Texts in Mathematics*.

4. **Preparata, F. P., & Shamos, M. I. (1985).** "Computational Geometry: An Introduction". *Springer-Verlag*.

5. **Dantzig, G. B. (1987).** "Origins of the Simplex Method". *Stanford University*.

6. **Strang, G. (2019).** "Linear Algebra and Learning from Data". *Wellesley-Cambridge Press*.

7. **de Berg, M. et al. (2008).** "Computational Geometry: Algorithms and Applications". *Springer*.

8. **Bentley, J. L. (1975).** "Multidimensional Binary Search Trees". *Communications of the ACM*.

9. **Friedman, J. H. (1991).** "Multivariate Adaptive Regression Splines (MARS)". *The Annals of Statistics*.

10. **Shannon, C. E. (1948).** "A Mathematical Theory of Communication". *Bell System Technical Journal*.

11. **Cybenko, G. (1989).** "Approximation by Superpositions of a Sigmoidal Function". *MCSS*.

12. **LeCun, Y., et al. (1989).** "Optimal Brain Damage". *NIPS*.

13. **Han, S., et al. (2015).** "Deep Compression: Compressing Deep Neural Networks". *arXiv:1510.00149*.

14. **Glorot, X., Bordes, A., & Bengio, Y. (2011).** "Deep Sparse Rectifier Neural Networks". *AISTATS*.

15. **Arora, S., et al. (2018).** "Understanding Deep Neural Networks with Rectified Linear Units". *ICLR*.

16. **He, K., et al. (2016).** "Deep Residual Learning for Image Recognition". *CVPR*.

17. **Rumelhart, D. E., et al. (1986).** "Learning representations by back-propagation errors". *Nature*.

18. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** "Deep Learning". *MIT Press*.

---

## PARTE 8

### 8. REPOSITORIOS

* **ATOM CORE:** [https://github.com/wexionar/slrm-atom-core](https://github.com/wexionar/slrm-atom-core)
* **LOGOS CORE:** [https://github.com/wexionar/slrm-logos-core](https://github.com/wexionar/slrm-logos-core)
* **LUMIN CORE:** [https://github.com/wexionar/slrm-lumin-core](https://github.com/wexionar/slrm-lumin-core)
* **NEXUS CORE:** [https://github.com/wexionar/slrm-nexus-core](https://github.com/wexionar/slrm-nexus-core)
* **LOGOS FUSION:** [https://github.com/wexionar/slrm-logos-fusion](https://github.com/wexionar/slrm-logos-fusion)
* **LUMIN FUSION:** [https://github.com/wexionar/slrm-lumin-fusion](https://github.com/wexionar/slrm-lumin-fusion)

---

- *Desarrollado para la comunidad global de desarrolladores. Cerrando la brecha entre la lógica geométrica y el modelado de alta dimensionalidad.*

- *Dos caminos divergían en el bosque, nosotros tomamos el menos transitado, y eso hizo que todo fuera diferente.*
 
