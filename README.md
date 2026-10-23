# Women Safety & Psychometric Analytics Platform (MVP 2.0)

A comprehensive analytics dashboard and research platform integrating **Women's Safety Geospatial Intelligence**, **Bharatiya Nyaya Sanhita (BNS) 2023 Legal Provisions**, and a rigorous **Psychometric Reliability & Confirmatory Factor Analysis (CFA) Engine**.

---

## 🚀 Quick Start

### 1. Environment & Setup (Windows / macOS / Linux)

Open a terminal in the project folder:

```bash
# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate
```

### 2. Launch Streamlit Application

```bash
streamlit run app.py
```

The application will be accessible at:
👉 **`http://localhost:8501`**

---

## 📦 Built-In Datasets Included

Out-of-the-box datasets are pre-loaded in `data/`:
1. **`data/women_safety_geospatial.csv`** (264 records across 8 major Indian metro regions with coordinates, safety scores, lighting, police patrol, CCTV coverage, and reported incidents).
2. **`data/bns_2023_sections.csv`** (25 key provisions focusing on Chapter V: Offences Against Women and Children, with punishments, bailability, cognizable status, and historic IPC equivalents).
3. **`data/psychometric_longitudinal_counselling.csv`** (1,000 observations for 500 participants across T0 Baseline and T1 Post-Counselling follow-ups, containing PHQ-9, GAD-7, Brief COPE, and standardized items).

You can also upload custom CSV/XLSX/JSON files directly through the application sidebar.

---

## 🛡️ Core Modules & Features

### 1. 📊 Executive Overview & Data Inventory
- Instant status cards of loaded datasets and total record volumes.
- Automatic column type inspection, non-null tracking, and sample previews.
- One-click dataset exports.

### 2. 🗺️ Women's Safety Geospatial Intelligence
- Interactive geospatial map with dynamic color scaling by safety index and incident severity.
- Filtering by City (Delhi, Mumbai, Bengaluru, Kolkata, Chennai, Hyderabad, Pune, Jaipur) and Risk Level.
- Boxplots and violin distributions showing the quantitative impact of lighting quality and police patrol frequency on neighborhood safety.

### 3. ⚖️ BNS 2023 Legal Explorer & IPC Converter
- **IPC-to-BNS Quick Converter**: Input historic IPC sections (e.g. 376, 354, 498A, 509, 302) to immediately display new BNS sections, definitions, and penalties.
- Full-text search across section numbers, titles, and legal provisions.
- Color-coded badges for Bailability (Bailable vs Non-Bailable), Cognizability, and Triable Courts.

### 4. 🧠 Psychometrics & CFA Engine
- **Reliability**: Cronbach's alpha ($\alpha$), McDonald's omega ($\omega$), and item-rest correlation with "alpha-if-item-deleted".
- **Item Correlation**: Pearson and Spearman correlation heatmaps.
- **Dimensionality**: Scree plot with Kaiser criterion ($\lambda \ge 1.0$) and Exploratory Factor Analysis (EFA).
- **Confirmatory Factor Analysis (CFA)**: Mathematical evaluation of $\chi^2$, degrees of freedom, CFI, TLI, RMSEA (with 90% confidence intervals), and SRMR with standard psychometric fit benchmarks.
- **Longitudinal Evaluation**: Paired Student's $t$-test, Wilcoxon signed-rank test, Cohen's $d$ effect sizes, and Pre vs Post boxplots.

### 5. 🤖 Machine Learning Predictive Modeling
- **Zero-Leakage Grouped Splits**: Employs `GroupShuffleSplit` on participant IDs to strictly prevent longitudinal data leakage between training and testing folds.
- **Regression & Classification**: Evaluates Linear Regression, Elastic Net, Random Forest, and Gradient Boosting.
- **Performance Diagnostics**: MAE, RMSE, $R^2$, actual vs. predicted regression charts, feature importance rankings, and classification metrics (Accuracy, F1, ROC-AUC).

### 6. 📜 Methodology, Ethics & Governance
- Methodological alignment with academic psychometric standards.
- Clear separation of domains to avoid unsubstantiated causal claims.
- Educational disclaimers regarding non-clinical use and legal lookup accuracy.

---

## 🛠️ Technology Stack
- **Frontend / Dashboard**: Streamlit (v1.66+), Custom CSS Glassmorphism
- **Visualization**: Plotly Express & Graph Objects (MapLibre / Carto)
- **Scientific Computing**: NumPy, Pandas, SciPy (Hypothesis Testing)
- **Machine Learning**: Scikit-Learn (FactorAnalysis, GroupShuffleSplit, Regressors, Classifiers)
- **Environment Tooling**: Python 3.13, uv
