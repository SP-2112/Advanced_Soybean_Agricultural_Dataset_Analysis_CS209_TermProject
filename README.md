# Advanced_Soybean_Agricultural_Dataset_Analysis
 


This project explores the clustering of soybean samples based on production quantity and quality metrics. We evaluate clustering quality using SHAP (SHapley Additive exPlanations) to interpret the importance of input features in defining cluster assignments. The goal is to identify meaningful groupings and understand what drives them.

### Dataset

The dataset (`3_duplicates_removed.csv`) contains key phenotypic traits of soybean samples, including:
- Sugars (Su)
- Weight of 300 Seeds (W3S)
- Seed Yield per Unit Area (SYUA)
- Number of Seeds per Pod (NSP)
- Protein Percentage (PPE)
- Protein Content (PCO)

---

### Key Analyses

- **Clustering (KMeans)** on both:
  - Original target-related features
  - PCA-reduced features
- **Cluster Evaluation** using average SHAP values across clusters
- **Interpretability**: SHAP Summary Plots show how each feature contributes to cluster labeling
- **Comparative Analysis**: Side-by-side evaluation of PCA vs original feature clustering

---

### SHAP-Based Evaluation Metric

To judge the "explainability" of cluster groups, we compute:
> 🔹 Average SHAP value of each feature per cluster  
> 🔹 Then average across all clusters  
> 🔹 The higher the SHAP impact, the more interpretable the clustering

---

### Installation & Usage

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run the Project

Open notebooks step-by-step inside the `/notebooks/` folder.  
Start with `01_data_preprocessing.ipynb`.

---

### Requirements

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
shap
gdown
```

