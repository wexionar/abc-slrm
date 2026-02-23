# ABC-SLRM: Segmented Linear Regression Model

**About the ABC Framework, its Foundations and its Relationship with SLRM**

**SLRM Team:**<br>
Alex · Gemini · ChatGPT<br>
Claude · Grok · Meta AI<br>
DeepSeek · GLM · Anna

**Version:** 2.0<br>
**License:** MIT<br>
**Date:** 2026-02

---

## PART 1

### 1. ABC FRAMEWORK

The ABC Framework defines three universal phases that every data modeling system must traverse.

### 1.1 Phase A: The Base Dataset

- **Definition:** A collection of N records in a D-dimensional space, where each record contains independent variables: X = [X_1, X_2, ... , X_D] and a dependent variable: Y. The functional relationship Y = f(X) is assumed.

- **Ideal:** Achieve a quasi-infinite and geometrically quasi-continuous dataset. 

- **Reference:** dataset.csv

### 1.2 Phase B: The Engines

The tools that transform and query the data. They are basically divided into three types of engines:

#### B.1 - Core Engines

- **Definition:** They infer over the Base Dataset.

- **Ideal:**  Infer as fast as possible over Base Dataset being as precise as possible.

- **Reference:** core.exe

#### B.2 - Origin Engines

- **Definition:** They optimize/compress the Base Dataset into Optimized Dataset.

- **Ideal:** Optimize/compress the Base Dataset as much as possible losing the minimum precision possible.

- **Reference:** origin.py

#### B.3 - Resolution Engines

- **Definition:** They infer over the Optimized Dataset.

- **Ideal:** Infer as fast as possible over Optimized Dataset being as precise as possible.

- **Reference:** resolution.exe

### 1.3 Phase C: The Optimized Dataset

- **Definition:** An Optimized/Compressed Dataset of the Base Dataset, with equal or different logical-mathematical structure.

- **Ideal:** Be able to represent all the data of the Base Dataset with the simplest logical-mathematical structure possible.

- **Reference:** dataset.npy

---

## PART 2

### 2. ABC MODELS

There are different models that follow the structure or anatomy of the ABC Framework.

### 2.1 The MML Model

- **Phase A:**<br>
└─ dataset.csv

- **Phase B:**<br>
├─ The Black Box<br>
└─ Et Cetera

- **Phase C:**<br>
└─ dataset.h5

### 2.2 The SLRM Model

- **Phase A:**<br>
└─ dataset.csv

- **Phase B.1:**<br>
├─ Atom Core<br>
├─ Logos Core<br>
├─ Lumin Core<br>
└─ Nexus Core

- **Phase B.2:**<br>
├─ Atom Origin<br>
├─ Logos Origin<br>
├─ Lumin Origin<br>
└─ Nexus Origin

- **Phase B.3:**<br>
├─ Atom Resolution<br>
├─ Logos Resolution<br>
├─ Lumin Resolution<br>
└─ Nexus Resolution

- **Phase C:**<br>
└─ dataset.npy

---

## PART 3

### 3. SLRM ENGINES

SLRM engines are organized according to the **point support** necessary to perform an inference. This hierarchy represents a progression of increasing complexity: from the simplicity of copying the nearest neighbor (Atom), to the exponential complexity of requiring 2^D points to build a complete polytope (Nexus).

### 3.1 The Three Types of Engines

- **B.1 - Core Engines**<br>
  ├─ Atom Core (support 1 point)<br>
  ├─ Logos Core (support 2 points)<br>
  ├─ Lumin Core (support D+1 points)<br>
  └─ Nexus Core (support 2^D points)

- **B.2 - Origin Engines**<br>
  ├─ Atom Origin (support 1 point)<br>
  ├─ Logos Origin (support 2 points)<br>
  ├─ Lumin Origin (support D+1 points)<br>
  └─ Nexus Origin (support 2^D points)
  
- **B.3 - Resolution Engines**<br>
  ├─ Atom Resolution (support 1 point)<br>
  ├─ Logos Resolution (support 2 points)<br>
  ├─ Lumin Resolution (support D+1 points)<br>
  └─ Nexus Resolution (support 2^D points)

### 3.2 The Ant Metaphor

- **Core Engines are Explorer Ants:** they discover how to infer, identify what structure they need, define what must be saved.

- **Origin Engines are Builder Ants:** they follow the path marked by Core, build the Optimized Dataset.

- **Resolution Engines are Worker Ants:** they use the constructed structure to infer efficiently.

### 3.3 The Fusion Architecture Analogy

- **Fusion** is an architecture that combines **two engines in a container.**

- Like a .tar file in Linux, **Fusion packages two engines** that work together:

  - Atom Fusion = AtomOrigin + AtomResolution<br>
  - Logos Fusion = LogosOrigin + LogosResolution<br>
  - Lumin Fusion = LuminOrigin + LuminResolution<br>
  - Nexus Fusion = NexusOrigin + NexusResolution

---

### 3.4 Engine Summary Chart

| Types of Engines | Core Engines | Origin Engines | Resolution Engines |
|:-----------------|:------------:|:--------------:|:-----------------:|
| **Support 1 Point** | Atom Core | Atom Origin | Atom Resolution |
| **Support 2 Points** | Logos Core | Logos Origin | Logos Resolution |
| **Support D+1 Points** | Lumin Core | Lumin Origin | Lumin Resolution |
| **Support 2^D Points** | Nexus Core | Nexus Origin | Nexus Resolution |
| **Core Container** | X | - | - |
| **Fusion Container** | - | X | X |

- **Atom** - The Searcher Specialist<br>
- **Logos** - The Unidimensional Specialist<br>
- **Lumin** - The Multidimensional Standard<br>
- **Nexus** - The Dense Grids Specialist

---

### 3.5 Comparative Table of Core Engines

| Engine | Support | Structure | Complexity | Relative Speed | Ideal Use |
|-------|---------|------------|-------------|-------------------|-----------|
| **Atom** | 1 point | Point | O(log N) | **1.0× (fastest)** ⚡ | Extremely dense datasets (N >> 10^6) |
| **Logos** | 2 points | Segment | O(log N) | **1.4× slower** | 1D time series |
| **Lumin** | D+1 points | Simplex | O(log N + D²) | **3.0× slower** | Standard nD datasets |
| **Nexus** | 2^D points | Polytope | O(log N + D log D) | **4-5× slower** | Structured grids (up to ~15D) |

**Note on Nexus:** Requires a complete grid with 2^D points. In high dimensions, this becomes exponentially unfeasible:
- 10D: 1,024 points (viable)
- 15D: 32,768 points (challenging)
- 20D: 1,048,576 points (very difficult)
- 50D: Practically impossible (>10^15 points)

---

### 3.6 Complexity Progression

The hierarchy Atom → Logos → Lumin → Nexus not only represents increase in support points, but also in computational complexity:

| Engine | Support Points | Search Complexity | Calculation Complexity | Structural Complexity | Relative Speed |
|-------|----------------|---------------------|-------------------|------------------------|-------------------|
| **Atom** | 1 | O(log N) | O(1) - direct copy | **Minimum** | **1.0× (fastest)** ⚡ |
| **Logos** | 2 | O(log N) | O(1) - 1D interpolation | **Low** | **1.4× slower** |
| **Lumin** | D+1 | O(log N) | O(D²) - linear system | **Moderate** | **3.0× slower** |
| **Nexus** | 2^D | O(log N) | O(D log D) - Kuhn partition | **Exponential** | **4-5× slower** |

---

## PART 4

### 4. SLRM QUALITY

#### 4.1 EPSILON (ε)
In the SLRM model, the Epsilon (ε) parameter is strictly defined based on the dependent variable (Y). SLRM uses ε as a tolerance threshold over the response (Y). 

**Quality Note:** This parameter determines the sensitivity of the model to variations in the function f(X) and acts as the main criterion for the simplification and linearization of the segments.

#### 4.2 INTERPOLATION
Inference process where the query point is located within the limits defined by the known data cloud.

**Practice Note:** In SLRM the quality of the inference depends on the data quality of the local sector to infer, therefore, it is established as good practice to verify the health of said sector before issuing a Y result.

#### 4.3 EXTRAPOLATION
Inference process where the query point is located outside the known limits of the Dataset for each coordinate or X axis.

**Honesty Note:** In SLRM, extrapolation is assumed as a theoretical projection. Since there is no physical data support that backs the query, the Y result is an extension of the local trend and must be treated with caution, recognizing that the linearity conjecture has no possible verification in this void.

---

## PART 5

### 5. GUARANTEES AND PHILOSOPHY

#### 5.1 Mathematical Guarantees

SLRM provides **two non-negotiable conditions**:

1. **Condition 1 (Training Precision):** Any point retained in the compressed model must be inferred with zero difference (or within epsilon) 
2. **Condition 2 (Precision of Discarded Points):** Any point discarded during compression must be inferred within epsilon, regardless of the input order.

These conditions ensure that model compression does not sacrifice precision.

#### 5.2 Design Philosophy

1. **Hardware Efficiency:** Designed for CPUs and microcontrollers. Inference eliminates GPU dependency. Training is efficient on CPU, although GPU acceleration is possible for large-scale ingestion.

2. **Total Transparency:** Every inference is traceable to the linear law of a specific sector. No hidden layers, no opaque transformations—only local geometry.

3. **The Simplex as an Atom:** Just as the derivative linearizes a curve at a point, the Simplex is the minimum unit that guarantees a pure linear law in n-dimensions without introducing artificial curvatures.

4. **Deterministic Logic:** No gradient descent, no stochastic optimization, no random initialization. Same dataset + same parameters = same model, always.

5. **Purpose:** To democratize high precision by eliminating the massive computational cost of gradient descent, returning data modeling to the field of deterministic logic.

---

## PART 6

### 6. CONCLUSION

SLRM represents a return to **fundamental geometric principles** in data modeling. By replacing gradient descent with deterministic partitioning, we achieve:

- **Transparency:** Every prediction is traceable
- **Efficiency:** Runs on CPUs and microcontrollers
- **Guarantees:** Error limited by epsilon, no hallucinations
- **Interpretability:** Linear laws with physical meaning

This is not a replacement for all neural networks, but a **rigorous alternative** for applications where transparency, efficiency, and determinism matter more than squeezing the last 0.1% of precision.

**The glass box is open.**

---

## PART 7

### 7. BIBLIOGRAPHY

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

## PART 8

### 8. REPOSITORIES

* **ATOM CORE:** [https://github.com/wexionar/slrm-atom-core](https://github.com/wexionar/slrm-atom-core)
* **LOGOS CORE:** [https://github.com/wexionar/slrm-logos-core](https://github.com/wexionar/slrm-logos-core)
* **LUMIN CORE:** [https://github.com/wexionar/slrm-lumin-core](https://github.com/wexionar/slrm-lumin-core)
* **NEXUS CORE:** [https://github.com/wexionar/slrm-nexus-core](https://github.com/wexionar/slrm-nexus-core)
* **LOGOS FUSION:** [https://github.com/wexionar/slrm-logos-fusion](https://github.com/wexionar/slrm-logos-fusion)
* **LUMIN FUSION:** [https://github.com/wexionar/slrm-lumin-fusion](https://github.com/wexionar/slrm-lumin-fusion)

---

- *Developed for the global community of developers. Closing the gap between geometric logic and high-dimensional modeling.*

- *Two paths diverged in the forest, we took the less traveled one, and that made everything different.*
 
