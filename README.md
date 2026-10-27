# Hospital Readmission Prediction

## Project title
**Hospital Readmission Prediction using Data Mining and Machine Learning**

## Recommended platform
- **VS Code** — development
- **Python 3.11+** — programming
- **scikit-learn** — machine learning
- **Pandas / NumPy** — data processing
- **Matplotlib / Seaborn** — visualization
- **Streamlit** — final web dashboard

## Dataset
This project is designed for the UCI **Diabetes 130-US Hospitals for Years 1999-2008** dataset.

Official source:
https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor-years%2B1999-2008

Download the dataset and place:
`diabetic_data.csv`
inside:
`dataset/diabetic_data.csv`

The target is the `readmitted` column:
- `<30` = readmitted within 30 days → target 1
- `>30` = not within 30 days → target 0
- `NO` = not readmitted → target 0

## Modules implemented
1. Dataset Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Classification Model Development
6. Model Evaluation and Comparison
7. Readmission Risk Prediction
8. Streamlit Web Dashboard

## Models
- Logistic Regression
- Decision Tree
- Random Forest

The training script evaluates all three and saves the best model based on F1-score.

## Project structure
```text
Hospital_Readmission_Prediction_DMT/
├── dataset/
│   └── diabetic_data.csv          <-- YOU ADD THIS
├── model/
│   └── model_bundle.joblib        <-- created after training
├── app/
│   └── app.py
├── notebooks/
│   └── hospital_readmission.ipynb
├── train_model.py
├── requirements.txt
└── README.md
```

## Setup in VS Code (Windows)

### 1. Create/activate a virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install packages
```powershell
pip install -r requirements.txt
```

### 3. Put the dataset in the correct location
```text
dataset/diabetic_data.csv
```

### 4. Train the models
From the project root:
```powershell
python train_model.py
```

This will:
- load the dataset
- clean missing values represented by `?`
- create the binary readmission target
- preprocess numerical/categorical features
- train Logistic Regression, Decision Tree and Random Forest
- print accuracy, precision, recall, F1-score and ROC-AUC
- save the best pipeline to `model/model_bundle.joblib`
- save `model/model_comparison.csv`
- save EDA plots in `model/plots/`

### 5. Run the Streamlit dashboard
```powershell
streamlit run app/app.py
```

The browser should open with the prediction dashboard.

## Important academic note
This is an educational machine-learning prototype, not a clinically validated medical decision system. Do not use it for real patient-care decisions.

## Suggested presentation explanation
> “Our system predicts whether a diabetic patient is likely to be readmitted within 30 days using historical hospital encounter data. We clean and preprocess the data, perform exploratory analysis and feature selection, train multiple classification algorithms, compare their performance using standard evaluation metrics, select the best model, and deploy it through a Streamlit prediction dashboard.”
