# Student Employability Prediction & Classification

This repository contains an end-to-end Machine Learning pipeline designed to predict and classify student employability status. By analyzing behavioral attributes, presentation performance, soft skills, and academic ratings, the model evaluates whether a student is categorized as **Employable** or **Less Employable**, providing a data-driven approach to career readiness assessment.

## Project Overview
Bridging the gap between graduation and employment is a major challenge in higher education. This project implements a standardized predictive framework to assess student profiles across multiple human-evaluative dimensions. Utilizing a normalized classification dataset, it handles preprocessing, feature scaling, training, optimization, and validation of a predictive model.

## Dataset Profile & Behavioral Features
The analysis is performed on student records containing key evaluative traits scored on structural performance scales:
* **GENERAL APPEARANCE**: Professional presentation, grooming, and dress code adherence.
* **MANNER OF SPEAKING**: Articulation, tone, clarity, and pacing of communication.
* **PHYSICAL CONDITION**: Energy levels, posture, and presence during evaluation.
* **MENTAL ALERTNESS**: Critical thinking speed, comprehension, and responsiveness to questioning.
* **SELF-CONFIDENCE**: Poise, emotional control, and assurance under pressure.
* **ABILITY TO PRESENT IDEAS**: Structured thought processing, logical flow, and persuasiveness.
* **COMMUNICATION SKILLS**: Overarching linguistic ability and engagement metrics.
* **Student Performance Rating**: Consolidated academic or behavioral benchmarks.
* **CLASS (Target Variable)**: The classification label indicating final job readiness (`Employable` vs. `Less Employable`).

## Machine Learning Pipeline

### 1. Data Cleaning & Encoding
- Isolates features (`X`) from target labels (`y`).
- Converts categorical target labels (`Employable` / `Less Employable`) into computer-readable numerical values.

### 2. Dataset Split
- Partitions the data into **Training (80%)** and **Testing (20%)** subsets to guarantee unbiased validation.

### 3. Feature Scaling (Data Normalization)
- Integrates `StandardScaler` from `scikit-learn` to scale behavioral ratings to a standard normal distribution (mean of 0 and variance of 1). This ensures features with varying variances do not distort model optimization.

### 4. Predictive Modeling
- Trains a **Logistic Regression** classifier on the scaled training matrix. Logistic Regression provides robust, highly interpretable probability scores for binary classification tasks.

### 5. Performance Validation
The model achieves well-balanced, high-accuracy results, demonstrating excellent generalization with no signs of overfitting:
* **Training Accuracy:** `84.86%`
* **Testing Accuracy:** `84.69%`

## Tech Stack & Libraries
- **Language:** Python
- **Data Engineering:** `pandas`, `numpy`
- **Machine Learning & Preprocessing:** `scikit-learn` (specifically `Logistic Regression`, `StandardScaler`, and `accuracy_score`)
- **Development Environment:** Jupyter Notebooks
