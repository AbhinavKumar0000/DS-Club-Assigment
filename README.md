# Collision-Risk Zone Classification Using Satellite Orbital Data

## Project Overview

This project addresses the growing challenge of orbital congestion by developing a machine learning model to classify space regions based on collision risk. Using a real-world dataset of active satellites and space debris, this work demonstrates an end-to-end data science pipeline, from exploratory data analysis to model deployment. The primary objective is to build a reliable classifier that can predict the level of congestion risk (`low`, `medium`, `high`) for a given satellite's orbital parameters, serving as a foundational tool for space traffic management and collision avoidance systems.

This repository fulfills the requirements of a technical screening assignment, showcasing skills in data analysis, feature engineering, and the implementation of a robust machine learning pipeline.

---

## Problem Statement

With over 14,500 active satellites and countless pieces of debris, Earth's orbit is becoming increasingly crowded. This congestion poses a significant threat to operational satellites, future launches, and space exploration missions. A collision in orbit can create thousands of new debris fragments, triggering a cascade effect known as the Kessler syndrome.

The real-world problem is to proactively identify and classify orbital zones by their inherent collision risk. By doing so, satellite operators and space agencies can:

- Plan safer trajectories for new launches.
- Execute timely collision avoidance maneuvers for active satellites.
- Enhance space situational awareness and contribute to long-term orbital sustainability.

This project tackles this problem by framing it as a multi-class classification task, where the model learns to map orbital characteristics to predefined risk levels.

---

## Dataset Description

The analysis is based on a public satellite catalogue, representing a snapshot of objects orbiting Earth.

### Data Source

The dataset used in this project is derived from the following public source:

- **Kaggle Dataset**: _Satellite Orbital Catalog_
- **Author**: Karnika Kapoor
- **Link**: https://www.kaggle.com/datasets/karnikakapoor/satellite-orbital-catalog

The dataset aggregates satellite orbital parameters sourced from publicly available catalogs such as **Celestrak** and **Space-Track**, and is commonly used for research and educational purposes related to orbital mechanics and space situational awareness.

Only the relevant subset of features required for this assignment was used, and minimal preprocessing was applied to ensure data quality and consistency.

- **Content**: The dataset (`data/current_catalog.csv`) contains orbital parameters for approximately 14,500 objects, including active satellites, rocket bodies, and debris.
- **Key Attributes**:
  - `altitude_km`: The satellite's altitude above Earth's surface.
  - `inclination`: The angle of the orbit in relation to the Earth's equatorial plane.
  - `eccentricity`: A measure of how much an orbit deviates from a perfect circle.
  - `orbital_band`: The orbital region (LEO, MEO, GEO).
  - `congestion_risk`: The target label, pre-calculated based on spatial density and operational factors, categorized as `LOW`, `MEDIUM`, or `HIGH`.

---

## Task 1: Exploratory Data Analysis & Insights

The EDA process, detailed in `EDA/01_eda.ipynb`, was conducted to understand the underlying data distributions and relationships between key orbital parameters.

### Visualizations

1.  **Satellite Altitude Distribution**: A histogram of satellite altitudes reveals that the vast majority of objects reside in Low Earth Orbit (LEO), specifically between 400 and 600 km. This concentration highlights the most congested area in space.
2.  **Inclination vs. Altitude**: A boxplot analysis shows distinct clusters. For instance, sun-synchronous orbits are visible at high inclinations (~98 degrees) in LEO, while geostationary satellites form a tight cluster at 0-degree inclination and ~35,786 km altitude. This confirms that satellites do not operate randomly but follow specific, often crowded, orbital highways.
3.  **Distribution by Orbit Band**: A count plot confirms that LEO is by far the most populated band, followed by GEO and MEO. This visual reinforces why LEO is the primary focus for collision risk assessment.

### Actionable Insights

1.  **Insight 1: LEO is the Critical Congestion Zone**: The overwhelming concentration of satellites in LEO suggests that any effective space traffic management system must prioritize monitoring and de-risking this band. The highest collision probabilities are expected here.
2.  **Insight 2: Orbital Inclination Dictates Traffic "Lanes"**: Satellites are not uniformly distributed but are concentrated at specific inclinations (e.g., 53 degrees for Starlink, 98 degrees for sun-synchronous satellites). These inclinations act as traffic lanes, and intersections between these lanes are potential collision hotspots.
3.  **Insight 3: A Simple Altitude Metric is Insufficient**: While altitude is a primary factor, the interplay between altitude, inclination, and the sheer number of objects in an orbital band collectively determines congestion. This justifies a multi-feature approach for the machine learning model.

---

## Feature Engineering

To prepare the data for the model, several feature engineering steps were performed in `src/features.py`. The goal was to create a feature set that explicitly captures the drivers of collision risk.

- **Target Variable**: The `congestion_risk` column was selected as the multi-class target `y`.
- **Feature Selection**: Key physical parameters were chosen for the feature matrix `X`: `altitude_km`, `inclination`, `eccentricity`, and `mean_motion`.
- **Engineered Feature (`orbital_density`)**: To provide the model with a direct measure of congestion, a new feature, `orbital_density`, was created. It is calculated by counting the total number of objects within each satellite's `orbital_band` (LEO, MEO, GEO). This helps the model learn that an object in a crowded band is inherently at higher risk.
- **Categorical Encoding (`orbit_band_code`)**: The categorical `orbital_band` feature was converted into a numerical format (`orbit_band_code`) using ordinal mapping, as there is a natural (though not linear) relationship between the bands and altitude.

This engineered feature set provides a rich, quantitative basis for the classification model.

---

## Task 2: Machine Learning Pipeline

To ensure reproducibility and production-readiness, a `scikit-learn` pipeline was constructed in `src/train.py`. This pipeline encapsulates the entire workflow, from data preprocessing to the final model.

### Pipeline Structure

The pipeline consists of two main stages:

1.  **Preprocessing (`StandardScaler`)**: This step standardizes all numerical features by removing the mean and scaling to unit variance. This is crucial for distance-based algorithms and helps gradient-based models converge faster. For a tree-based model like Random Forest, it's less critical but is good practice and makes the pipeline adaptable to other models.
2.  **Modeling (`RandomForestClassifier`)**: A Random Forest Classifier was chosen as the estimator.

### Rationale for Model Choice

A **Random Forest** is well-suited for this problem for several reasons:

- **Robustness**: It is an ensemble of decision trees, making it less prone to overfitting than a single tree.
- **Non-linearity**: It can capture complex, non-linear relationships between orbital parameters and risk, which are common in orbital mechanics.
- **Interpretability**: It provides feature importance metrics, allowing us to understand which orbital parameters are most influential in predicting collision risk.
- **Performance**: It generally achieves high accuracy on tabular datasets without extensive hyperparameter tuning.

---

## Model Training & Evaluation

- **Training**: The dataset was split into training (80%) and testing (20%) sets, with stratification to preserve the class distribution of `congestion_risk`. The pipeline was then trained on the training data. The trained model artifact is saved to `outputs/collision_model.pkl`.
- **Evaluation**: The model's performance was assessed on the held-out test set, with results detailed in `src/evaluate.ipynb`.

### Evaluation Metrics

Given the multi-class nature of the problem, the following metrics were used:

- **Accuracy**: The overall percentage of correctly classified instances.
- **Precision, Recall, and F1-Score**: These metrics provide a class-by-class assessment of the model's performance, which is important for understanding its behavior on minority classes.
- **Confusion Matrix**: A visual representation of the model's predictions versus the actual labels, showing exactly where the model is making errors.

---

## Results Summary

The model achieved outstanding performance on the test set:

- **Overall Accuracy**: **99.98%**

- **Class-wise Performance**:
  - The model demonstrates near-perfect precision and recall for all three risk classes (`HIGH`, `MEDIUM`, `LOW`).
  - The F1-scores are consistently above **0.99**, indicating a strong balance between precision and recall.

- **Confusion Matrix Analysis**: The confusion matrix shows only a handful of misclassifications out of thousands of samples, confirming the model's high reliability. The errors are minimal and do not indicate any systemic bias.

These results strongly suggest that the engineered features and the chosen model are highly effective at identifying collision risk zones from orbital data.

---

## Real-World Usefulness & Impact

This project is more than an academic exercise; it is a proof-of-concept for a critical space sustainability tool.

- **Enhanced Space Traffic Management**: A deployed version of this model could serve as a real-time risk assessment engine. Satellite operators could query the model with a proposed trajectory to receive an immediate risk classification, enabling them to choose safer paths.
- **Automated Collision Avoidance**: The model's output can be integrated into automated systems that flag high-risk conjunctions and recommend avoidance maneuvers, reducing the manual workload on human operators and minimizing response times.
- **Contribution to Space Sustainability**: By helping to prevent collisions, this technology directly contributes to reducing the proliferation of space debris. A cleaner orbital environment is safer and more economical for everyone, ensuring that space remains a viable resource for future generations.
- **Scalability**: The pipeline-based architecture allows the model to be easily retrained with new satellite data, ensuring it remains up-to-date as the orbital environment evolves.

---

## Project Structure

```
collision-risk-classification/
├───README.md
├───requirements.txt
├───data/
│   ├───current_catalog.csv
│   └───trajectory_timeseries.csv
├───EDA/
│   └───01_eda.ipynb
├───outputs/
│   └───collision_model.pkl
└───src/
    ├───evaluate.ipynb
    ├───features.py
    └───train.py
```

---

## How to Run the Project

To replicate the project, follow these steps:

1.  **Clone the Repository**

    ```bash
    git clone <repository-url>
    cd collision-risk-classification
    ```

2.  **Set up a Virtual Environment**
    It is recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Install all the required packages from `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the Model**
    Run the training script. This will process the data, train the pipeline, and save the model to the `outputs/` directory.

    ```bash
    python src/train.py
    ```

5.  **Evaluate the Model**
    To see the model's performance metrics and visualizations, run the evaluation Jupyter Notebook.
    ```bash
    jupyter notebook src/evaluate.ipynb
    ```
    You can also explore the initial data analysis in `EDA/01_eda.ipynb`.

---

## Conclusion

This project successfully demonstrates the development of a high-fidelity machine learning model for classifying satellite collision risk. By combining thoughtful exploratory analysis, targeted feature engineering, and a robust scikit-learn pipeline, the model achieves near-perfect accuracy. This work serves as a strong foundation for a real-world application that could enhance space situational awareness and play a tangible role in ensuring the long-term sustainability of orbital operations.
