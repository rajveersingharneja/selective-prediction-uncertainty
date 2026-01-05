# Selective Prediction under Uncertainty
## When Should a Machine Learning Model Refuse to Predict?

## Overview
This project explores selective prediction, a reliability-aware machine learning paradigm in which a model is allowed to abstain from making a prediction when its confidence is low.

Instead of forcing automated decisions for every input, the system selectively defers uncertain cases for human review, reducing overall risk in high-stakes decision environments.

The focus is not on maximizing accuracy, but on designing trustworthy and safe ML systems that know when not to decide.

## Motivation
In real-world deployments such as finance, healthcare, and autonomous systems:
- Incorrect automated decisions can be extremely costly
- Low-confidence predictions increase operational and ethical risk
- Human intervention is preferable for ambiguous cases

Traditional classifiers are not designed to say “I don’t know”.
This project explicitly models abstention as a valid third action, enabling safer deployment.

## Problem Formulation
Instead of binary classification, the model has three possible actions:
- Predict 0: Confident negative prediction
- Predict 1: Confident positive prediction
- Abstain: Low confidence, defer to human

The objective is to balance:
- Coverage (fraction of samples predicted automatically)
- Risk (error rate on predicted samples)
- Cost of wrong decisions versus abstention

## Methodology
1. Trained a probabilistic Logistic Regression classifier
2. Used predicted probabilities as confidence estimates
3. Defined a confidence band for selective prediction
4. Abstained on low-confidence samples
5. Evaluated performance using coverage, selective accuracy, and risk–coverage curves
6. Optimized abstention policy using cost-aware decision rules

## Key Insights
- Allowing abstention significantly reduces decision risk
- Predicting fewer cases can lead to more reliable outcomes
- Optimal automation depends on the cost of errors versus human review
- ML systems should explicitly model uncertainty rather than hide it

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Results
The project demonstrates:
- Selective prediction with a reject option
- Risk–coverage trade-offs under different confidence thresholds
- Cost-aware abstention policies for high-risk decision systems

A risk–coverage curve is generated to visualize the relationship between automation level and error rate.

## How to Run

1. Clone the repository:
git clone https://github.com/<your-username>/selective-prediction-uncertainty.git
cd selective-prediction-uncertainty

2. Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

(Windows)
venv\\Scripts\\activate

3. Install dependencies:
pip install -r requirements.txt

4. Add dataset:
Place the dataset CSV file inside:
data/raw/

Example:
data/raw/credit_default.csv

Note: The dataset is not included in this repository to avoid data licensing issues.

5. Launch the notebook:
jupyter notebook notebooks/01_selective_prediction.ipynb

6. Run all cells sequentially to reproduce the analysis and results.

## Future Work
- Bayesian uncertainty estimation
- Calibration-aware abstention strategies
- Deep learning extensions
- Application to healthcare and autonomous systems

## Author
Rajveer Arneja
