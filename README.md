#  Traffic Volume Prediction using Machine Learning & Deep Learning

##  Project Overview

This project predicts **traffic volume** based on weather conditions, time, and environmental factors using multiple **Machine Learning and Deep Learning models**.

The system compares several models such as **Linear Regression, Random Forest, XGBoost, SVR, KNN, and LSTM**, selects the best performing one, and allows users to make predictions through an **interactive Streamlit web application**.

The goal of the project is to analyze how different factors like **weather, time of day, and holidays influence traffic congestion**.

---

#  Technologies Used

### Programming

* Python

### Machine Learning

* Scikit-Learn
* XGBoost

### Deep Learning

* TensorFlow / Keras (LSTM)

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Web Application

* Streamlit

---

#  Models Implemented

The project trains and compares several models:

| Model                           | Type                        |
| ------------------------------- | --------------------------- |
| Linear Regression               | Machine Learning            |
| Decision Tree                   | Machine Learning            |
| Random Forest                   | Machine Learning            |
| K-Nearest Neighbors (KNN)       | Machine Learning            |
| Support Vector Regression (SVR) | Machine Learning            |
| XGBoost                         | Gradient Boosting           |
| LSTM                            | Deep Learning (Time-Series) |

The model with the **lowest Mean Absolute Error (MAE)** is selected as the best model.

---

#  Project Structure

```
ML-TRAFFIC-PREDICTION
│
├── data
│   └── Datafile.csv
│
├── images
│   ├── model_comparison.png
│   └── visualization.png
│
├── models
│   ├── lstm_model.h5
│   ├── traffic_model.pkl
│   └── model_columns.pkl
│
├── train_models.py
├── app.py
├── traffic_prediction.ipynb
├── requirements.txt
└── README.md
```

---

#  Features Used for Prediction

The model uses the following input features:

* Temperature
* Rainfall in last hour
* Snowfall in last hour
* Cloud coverage
* Hour of the day
* Day of the month
* Month
* Weekday
* Holiday indicator
* Weather conditions

These features help the model understand **patterns affecting traffic volume**.

---

#  Model Training

The training script:

```
train_models.py
```

Performs the following steps:

1. Loads the dataset
2. Preprocesses the data
3. Converts categorical variables using **One-Hot Encoding**
4. Splits data into training and testing sets
5. Trains multiple machine learning models
6. Evaluates models using **Mean Absolute Error (MAE)**
7. Saves the best model for prediction

---

# Model Comparison

The project generates a visualization comparing the performance of each model.

Example output:

* **Random Forest**
* **XGBoost**
* **SVR**
* **KNN**
* **LSTM**

The results are stored in:

```
images/model_comparison.png
```

---

# 🌐 Streamlit Web Application

A **Streamlit web app** allows users to interactively predict traffic volume.

Users can input:

* Temperature
* Rain / Snow
* Cloud percentage
* Time of day
* Holiday status
* Weather conditions

The model then predicts **expected traffic volume**.

---

# ▶️ How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/ml-traffic-prediction.git
cd ml-traffic-prediction
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Train the models

```
python train_models.py
```

This will create the trained models inside the `models` folder.

### 4️⃣ Run the Streamlit app

```
streamlit run app.py
```

Then open the browser at:

```
http://localhost:8501
```

---

#  Dataset

The dataset contains information about **traffic conditions and weather data**.

Key attributes include:

* Weather conditions
* Temperature
* Rain and snow levels
* Time and date
* Traffic volume

---

#  Future Improvements

Possible improvements for the project:

* Deploy the Streamlit app online
* Improve LSTM time-series modeling
* Add real-time traffic API integration
* Implement hyperparameter tuning
* Add traffic prediction visualization dashboard

---

#  Author

Machine Learning Project built for learning and experimentation with:

* Machine Learning
* Deep Learning
* Model comparison
* Interactive ML applications
