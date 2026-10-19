# 🏥 Triaje Médico Predictivo y Detección Temprana de Depresión con Big Data (CDC BRFSS 2024)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bernal-XFK/BigData-Medical-Triage-Depression-BRFSS/blob/main/Taller_BigData_Triaje_Medico_Depresion_BRFSS.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CDC_BRFSS_2024_(N=457k)-00629B.svg)](https://www.cdc.gov/brfss/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/Universidad-CESMAG-green.svg)](https://www.unicesmag.edu.co/)

> **Taller Práctico 02: Aprendizaje Automático con Big Data**  
> **Especialización en Big Data — Universidad CESMAG**  
> **Autor:** Jose Alejandro Bernal Figueroa ([@Bernal-XFK](https://github.com/Bernal-XFK))  
> **Metodología:** CRISP-DM (*Cross-Industry Standard Process for Data Mining*)  
> **Condición Médica de Estudio:** Trastorno Depresivo diagnosticado (`ADDEPEV3` — *Depressive Disorder*)

---

## 📌 Resumen Ejecutivo del Proyecto

El trastorno depresivo mayor representa una de las principales causas mundiales de discapacidad y carga de enfermedad en salud pública (OMS). En la práctica clínica y comunitaria, la detección temprana enfrenta barreras críticas: el estigma social, la escasez de psiquiatras y psicólogos en atención primaria, y la limitación de tiempo en las consultas rutinarias.

Este proyecto desarrolla e implementa un **sistema de triaje médico predictivo y tamizaje inteligente a escala poblacional**, capaz de estimar el riesgo de depresión en pacientes a partir de **40 variables multidimensionales** (salud física, limitaciones funcionales, hábitos de sueño, actividad física, comorbilidades crónicas, barreras socioeconómicas y sobrecarga de cuidado), **excluyendo intencionalmente cualquier variable tautológica de salud mental** para evitar fugas de información (*target leakage*).

El desarrollo se fundamenta en el sistema de vigilancia epidemiológica más grande del mundo: el **BRFSS 2024 (*Behavioral Risk Factor Surveillance System*) de los CDC (*Centers for Disease Control and Prevention*)**, con más de **457,000 registros poblacionales**, contrastando los hallazgos directamente contra la investigación de referencia de la Universidad de Columbia (*Yoon et al., 2015 — PMC4580372*).

---

## ⚖️ El Problema del Triaje Médico y la Asimetría de Costos

En el diseño de sistemas de inteligencia artificial aplicados a la medicina, la métrica tradicional de **Exactitud Global (*Accuracy*) es engañosa e inútil**:
* En una población con prevalencia de depresión del **21.1%** (desbalance 1 : 3.73), un modelo ingenuo que prediga siempre que el paciente está *"Sano"* obtendría un aparente **~79% de exactitud**, pero tendría una **Sensibilidad (*Recall*) del 0%**, dejando al 100% de las personas con depresión sin diagnóstico ni atención.

En un triaje clínico, los errores tienen **costos asimétricos extremos**:
1. **Falso Negativo (FN — Error Tipo II):** El modelo clasifica a una persona con depresión como sana. **Costo catastrófico:** el paciente no recibe tratamiento, el cuadro empeora, aumenta el riesgo de cronificación, deterioro laboral/familiar, y en casos severos, ideación suicida.
2. **Falso Positivo (FP — Error Tipo I):** El modelo activa una alerta preventiva para una persona sana. **Costo moderado/manejable:** la persona es derivada a una evaluación confirmatoria con psicología o una entrevista clínica estructurada (PHQ-9 / HAM-D), descartándose el caso sin daño físico.

Por esta razón, la arquitectura del proyecto se diseñó para **maximizar la Sensibilidad clínica y el ROC-AUC**, optimizando los umbrales de decisión mediante el **Índice de Youden ($J = \text{Sensibilidad} + \text{Especificidad} - 1$)** y aplicando esquemas de aprendizaje sensible al costo (*Cost-Sensitive Learning*).

---

## 🔄 Metodología CRISP-DM Aplicada

```
  ┌─────────────────────────┐
  │ 1. Negocio / Clínica    │  Definición del triaje, asimetría FN vs FP, target ADDEPEV3
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │ 2. Entendimiento Datos  │  CDC BRFSS 2024 (457,670 filas, 335 cols), prevalencia 21.1%
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │ 3. Preparación / ETL    │  Anti-leakage, selección de 40 predictores, ColumnTransformer
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │ 4. Modelado Supervisado │  5-Fold Stratified CV: Regresión Logística L2, RF, AdaBoost
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │ 5. Evaluación Clínica   │  Test Set independiente, Youden's J, Matrices de Confusión, ORs
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │ 6. Despliegue y Límites │  Contraste Yoon et al., Techo de Datos, Recomendaciones Salud
  └─────────────────────────┘
```

### 1. Comprensión del Negocio y Objetivo Clínico
* **Target:** `ADDEPEV3` (*"Has a doctor, nurse, or other health professional EVER told you that you have a depressive disorder (including depression, major depression, dysthymia, or minor depression)?"*).
* Codificación estricta: `1 = Depresión diagnosticada` (Positivo), `0 = Sin diagnóstico` (Negativo). Registros con valores nulos, códigos 7 (no sabe) o 9 (se negó a responder) fueron depurados conforme a la guía oficial de los CDC.

### 2. Entendimiento y Exploración de Datos (EDA)
* Población total BRFSS 2024: **457,670 encuestados**.
* Respuestas diagnósticas válidas en target: **455,006 encuestados (99.42%)**.
* Desbalance poblacional: **96,047 positivos (21.11%)** vs. **358,959 negativos (78.89%)** — Razón $1 : 3.73$.
* Tamaño de muestra representativa para modelado riguroso: **100,000 registros estratificados** para garantizar estabilidad matemática, velocidad de cómputo y convergencia de remuestreo.

### 3. Preparación de Datos y Prevención Rigurosa de *Target Leakage*
Para garantizar que el modelo aprenda señales biológicas, conductuales y contextuales reales y no relaciones circulares o tautológicas:
* **Exclusión estricta de variables de salud mental directa:** `MENTHLTH` (días con mala salud mental), `_MENT14D` (categoría de salud mental), `SDLONELY` (frecuencia de soledad), `LSATISFY` (satisfacción con la vida) y `EMTSUPRT` (apoyo emocional).
* **Selección parsimoniosa de 40 variables predictoras** clasificadas en 8 dimensiones clínicas:
  1. *Salud Física y Discapacidad Funcional:* `PHYSHLTH`, `_PHYS14D`, `POORHLTH` (días en que la mala salud impidió actividades habituales).
  2. *Disfunción Cognitiva y Sensorial:* `DECIDE` (dificultad para concentrarse, recordar o tomar decisiones), `BLIND`, `DIFFWALK`, `DIFFDRES`, `DIFFALON`.
  3. *Sueño y Descanso:* `SLEPTIM1` (horas de sueño habitual).
  4. *Enfermedades Crónicas e Inflamatorias:* `_DRDXAR2` (artritis), `CHCCOPD3` (EPOC), `ASTHMA3`, `CVDINFR4` (infarto), `CVDCRHD4` (enfermedad coronaria), `CVDSTRK3` (ACV), `CHCSCNC1` / `CHCOCNC1` (cáncer), `CHCKDNY2` (enfermedad renal), `DIABETE4` (diabetes).
  5. *Factores Conductuales y Estilo de Vida:* `_TOTINDA` (actividad física), `SMOKDAY2`, `_RFSMOK3`, `ALCDAY4`, `DRNKANY6`, `_RFBING6`.
  6. *Salud General y Parámetros Antropométricos:* `GENHLTH`, `_BMI5CAT` (categorías de IMC), `WTKG3`.
  7. *Factores Sociodemográficos y Acceso:* `_AGEG5YR`, `_SEX`, `_EDUCAG`, `_INCOMG1`, `EMPLOY1`, `PRIMINS2` (tipo de seguro), `MEDCOST1` (barrera económica para atención médica), `PERSDOC3`, `CHECKUP1`.
  8. *Sobrecarga del Cuidador:* `CRGVPRB4` (problema de salud del familiar cuidado).
* **Pipeline de Preprocesamiento (`ColumnTransformer`):**
  * Continuas: Imputación por mediana + escalado robusto (`RobustScaler`).
  * Ordinales: Imputación por mediana + estandarización (`StandardScaler`).
  * Categóricas/Nominales: Imputación explícita de categoría ausente (`-1`) para módulos opcionales (e.g. cuidadores) + codificación One-Hot (`OneHotEncoder(drop='first', handle_unknown='ignore')`).

---

## 🔬 Modelado y Validación Cruzada (5-Fold Stratified CV)

Se entrenaron y validaron tres familias complementarias de algoritmos utilizando **Validación Cruzada Estratificada de 5 Pliegues** sobre el 80% de entrenamiento ($N=80,000$):
1. **Regresión Logística con Regularización L2 (`class_weight='balanced'`):** Modelo lineal paramétrico de alta interpretabilidad clínica directa mediante *Odds Ratios*.
2. **Random Forest Classifier (`class_weight='balanced_subsample'`):** Ensamble por agregación (*Bagging*) no lineal con 150 árboles, capaz de capturar interacciones de orden superior sin riesgo de sobreajuste.
3. **AdaBoost Classifier con Estimadores Base Balanceados:** Ensamble por potenciación secuencial adaptativa (*Boosting*), calibrado con árboles de profundidad restringida y pesos de penalización clínica (inspirado en la arquitectura de Yoon et al., 2015).

### Resultados de la Validación Cruzada (5-Fold Stratified CV):
| Modelo Evaluado | ROC-AUC Medio | Desv. Est. ROC-AUC | Recall Medio | F1-Macro Medio | Balanced Accuracy Medio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística (L2 Balanced)** | **0.7873** | $\pm 0.0032$ | **0.6620** | **0.6730** | **0.7135** |
| **Random Forest (Balanced Subsample)** | **0.7912** | $\pm 0.0054$ | **0.6420** | **0.6834** | **0.7159** |
| **AdaBoost (Yoon et al. 2015)** | **0.7895** | $\pm 0.0047$ | 0.2970* | 0.6547 | 0.6301 |

> *\*Nota:* AdaBoost con umbral por defecto (0.50) sufre de bajo recall debido al desbalance natural de probabilidades empíricas de los stumps, defecto que se resuelve radicalmente mediante la calibración de umbral de Youden.

---

## 🎯 Desempeño en Test Set Independiente y Calibración de Youden

Evaluación final ejecutada sobre el conjunto de prueba independiente ($X_{\text{test}}$, $N=20,000$), sin contaminar el entrenamiento:

### 1. Desempeño Comparativo en Test Set ($X_{\text{test}}$):
| Modelo | ROC-AUC | PR-AUC | Sensibilidad (Recall) | Precisión | F1-Score | Balanced Accuracy | Accuracy Global |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced)** | **0.8069** | **0.5744** | **0.6674** | 0.4551 | **0.5504** | **0.7322** | 77.73% |
| **Regresión Logística (L2 Balanced)** | **0.7991** | **0.5655** | **0.7065** | 0.4404 | 0.5376 | **0.7272** | 74.93% |
| **AdaBoost (Threshold Calibrado Youden)** | 0.7631 | 0.5412 | **0.7012** | 0.4420 | 0.5416 | 0.7152 | 76.58% |

---

### 2. Impacto de la Calibración del Umbral Clínico (Índice de Youden $J$)
El Índice de Youden maximiza la distancia vertical respecto a la línea de no discriminación:
$$J = \max_{\tau} \left( \text{Sensibilidad}(\tau) + \text{Especificidad}(\tau) - 1 \right)$$

| Configuración de Modelo | Umbral de Corte ($\tau$) | Sensibilidad (Recall) | Especificidad | F1-Score | Balanced Accuracy | Falsos Negativos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost (Umbral Default 0.50)** | 0.5000 | 29.41% | 96.58% | 0.4139 | 0.6300 | 2,979 |
| **AdaBoost (Umbral Youden Calibrado)** | **0.3595** | **65.48%** | **79.56%** | **0.5416** | **0.7252** | **1,457 (Reducción >51%)** |
| **Regresión Logística Calibrada** | **0.4806** | **70.65%** | **74.79%** | **0.5376** | **0.7272** | **1,239** |
| **Random Forest Calibrado** | **0.4886** | **66.74%** | **79.71%** | **0.5504** | **0.7322** | **1,404** |

> **Conclusión de Triaje:** Mover el umbral de decisión permite reducir los falsos negativos a más de la mitad, rescatando a miles de pacientes en riesgo sin degradar inaceptablemente la especificidad (~75-80%).

---

## 🧠 Descubrimientos Clínicos y Factores de Mayor Impacto

El cruce de la importancia de características en modelos de ensamble (Reducción de Impureza de Gini) con los coeficientes paramétricos (*Odds Ratios* de Regresión Logística) revela patrones epidemiológicos de alto valor:

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                         TOP FACTORES PREDICTORES                            │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ 1. POORHLTH (Incapacidad funcional diaria)     ──► Gini #1 en RF (~0.155)   │
  │ 2. DECIDE = Sí (Dificultad cognitiva/memoria)  ──► OR = 1.82 - 3.73 (p<0.001)│
  │ 3. _SEX = Mujer (Vulnerabilidad biológica)     ──► OR = 1.50 - 1.98 (p<0.001)│
  │ 4. PRIMINS2 / MEDCOST1 (Barrera económica)     ──► OR = 1.31 - 1.51 (p<0.001)│
  │ 5. CRGVPRB4 (Sobrecarga del cuidador)          ──► OR = 1.45 (p<0.001)      │
  │ 6. _DRDXAR2 (Artritis y dolor crónico)         ──► OR = 1.30 (p<0.001)      │
  │ 7. CHCCOPD3 (EPOC / Enfermedad respiratoria)   ──► OR = 1.30 (p<0.001)      │
  └─────────────────────────────────────────────────────────────────────────────┘
```

1. **Incapacidad Funcional en Actividades Habituales (`POORHLTH`):**
   * Es el predictor más potente en Random Forest (Gini $\approx 0.155$). Cuanto mayor es el número de días en que la salud física o emocional impide trabajar, estudiar o socializar, mayor es la probabilidad de depresión clínica.
2. **Disfunción Cognitiva / Ejecutiva (`DECIDE_1.0`):**
   * Presentar dificultad subjetiva para recordar, concentrarse o tomar decisiones eleva significativamente los *odds* de depresión ($\text{OR} \approx 1.82 - 3.73$), reflejando el compromiso neurobiológico de la corteza prefrontal en estados depresivos.
3. **Diferencial de Sexo Biológico (`_SEX_1.0`):**
   * Ser mujer se asocia a un incremento de $\approx 1.50 - 1.98$ veces en la probabilidad diagnóstica de depresión, replicando hallazgos epidemiológicos globales sobre factores hormonales, sociales y de diagnóstico.
4. **Barreras de Acceso y Estrés Económico (`MEDCOST1`, `PRIMINS2`):**
   * No haber podido consultar al médico en el último año debido a limitaciones de dinero incrementa el riesgo de depresión en más de un 31% ($\text{OR} \approx 1.31$), evidenciando el nexo entre precariedad económica y sufrimiento psicológico.
5. **Comorbilidades Físicas Crónicas y Dolor (`_DRDXAR2`, `CHCCOPD3`):**
   * Padecer artritis o enfermedades inflamatorias crónicas incrementa en $\approx 1.30$ los *odds* de depresión. El dolor crónico y la restricción de movilidad actúan como catalizadores depresivos persistentes.
6. **Sobrecarga del Cuidador (`CRGVPRB4_5.0`):**
   * Asumir el cuidado no remunerado de un familiar enfermo o dependiente eleva el riesgo en un 45% ($\text{OR} \approx 1.45$), validando el síndrome de desgaste o *burnout* del cuidador.

---

## 📖 Contraste Directo con la Literatura (*Yoon et al., 2015*)

Este trabajo se evaluó rigurosamente frente al estudio de referencia de la **Universidad de Columbia**:
> *Yoon J, et al. (2015). Predicting depression from the Behavioral Risk Factor Surveillance System data using machine learning algorithms. AMIA Annu Symp Proc, 2015: 1346–1355. PMCID: PMC4580372.*

| Dimensión de Análisis | Estudio Original (*Yoon et al., 2015*) | Nuestro Desarrollo (BRFSS 2024) | Veredicto y Diferencial Técnico |
| :--- | :---: | :---: | :--- |
| **Poder Discriminativo (ROC-AUC)** | `0.7000 – 0.7200` | **`0.8069` (RF) / `0.7991` (Logística)** | **Supera ampliamente el benchmark.** Incremento superior a **+0.08** en ROC-AUC sin recurrir a variables de fuga. |
| **Gestión del Desbalance de Clases** | Sin balanceo explícito; dependiente de *Accuracy* global. | **`class_weight='balanced'` + Calibración de Youden ($J$).** | Se corrigió el sesgo de clase mayoritaria, elevando el Recall del 29% a **70.65%**. |
| **Prevención de *Data Leakage*** | Uso parcial de variables de salud mental. | **Exclusión total de variables tautológicas (`MENTHLTH`, `SDLONELY`).** | El modelo aprende exclusivamente a partir de proxies físicos, somáticos y socioeconómicos. |
| **Identificación del Predictor Principal** | Limitación funcional (*usual activity limitation*) y artritis. | **`POORHLTH` (#1 en RF), `DECIDE` (#1 OR) y `_DRDXAR2`.** | **Confirmación y validación empírica:** La incapacidad funcional diaria sigue siendo la señal más fidedigna. |

---

## 🧱 El Techo de los Datos y Límites del Triaje Automatizado

A pesar de alcanzar un **ROC-AUC > 0.80** y una **Sensibilidad del ~71%**, el modelo enfrenta un **techo intrínseco en los datos (*Data Ceiling*)**:
1. **Naturaleza Transversal de la Encuesta:** El BRFSS es un estudio de corte transversal (*cross-sectional*); no permite establecer causalidad temporal estricta (¿la limitación funcional causó la depresión o la depresión causó la incapacidad?).
2. **Sesgo de Autorreporte:** La variable objetivo `ADDEPEV3` mide diagnóstico previo por un profesional. En poblaciones vulnerables sin acceso a servicios de salud, muchas personas con depresión severa nunca han recibido un diagnóstico formal, siendo etiquetadas erróneamente como "negativas" en la encuesta.
3. **Ausencia de Biomarcadores y Genética:** El BRFSS no recopila parámetros biológicos directos (perfil tiroideo, niveles de cortisol, neuroimágenes o polimorfismos genéticos), los cuales explican la varianza biológica restante.
4. **Uso como Herramienta de Apoyo, Nunca de Reemplazo:** Este sistema debe emplearse estrictamente como un **mecanismo de alerta temprana y triaje poblacional**, que asista a médicos generales y enfermeros para formular tamizajes clínicos estructurados (PHQ-9), pero jamás como un diagnóstico definitivo automatizado.

---

## 🚀 Cómo Ejecutar el Proyecto

### Opción 1: En la Nube con Google Colab (Recomendada)
Haz clic directamente en el siguiente enlace para abrir el notebook interactivo con todas las gráficas y salidas precomputadas:
👉 [**Abrir en Google Colab**](https://colab.research.google.com/github/Bernal-XFK/BigData-Medical-Triage-Depression-BRFSS/blob/main/Taller_BigData_Triaje_Medico_Depresion_BRFSS.ipynb)

*El notebook incluye un script automatizado que descarga, extrae y procesa los datos del BRFSS 2024 directamente desde los servidores de los CDC en caso de no disponer del archivo local.*

---

### Opción 2: Ejecución Local en tu Entorno de Desarrollo
1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Bernal-XFK/BigData-Medical-Triage-Depression-BRFSS.git
   cd BigData-Medical-Triage-Depression-BRFSS
   ```

2. **Crear y activar un entorno virtual de Python:**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux / macOS:
   source venv/bin/activate
   ```

3. **Instalar dependencias requeridas:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Iniciar Jupyter Notebook:**
   ```bash
   jupyter notebook Taller_BigData_Triaje_Medico_Depresion_BRFSS.ipynb
   ```

---

## 📁 Estructura del Repositorio

```
BigData-Medical-Triage-Depression-BRFSS/
│
├── Taller_BigData_Triaje_Medico_Depresion_BRFSS.ipynb  # Notebook interactivo principal ejecutado
├── requirements.txt                                    # Librerías y versiones de Python requeridas
├── .gitignore                                          # Exclusión de archivos pesados de datos y caché
├── LICENSE                                             # Licencia MIT con Atribución Académica
└── README.md                                           # Documentación metodológica y reporte técnico
```

---

## 📜 Licencia y Uso Académico

Este proyecto se distribuye bajo la licencia **MIT con Atribución Académica y Educativa**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

Cualquier uso derivado, reproducción académica o referencia en investigaciones debe otorgar los créditos correspondientes a:
* **Autor:** Jose Alejandro Bernal Figueroa
* **Institución:** Universidad CESMAG — San Juan de Pasto, Nariño, Colombia.
