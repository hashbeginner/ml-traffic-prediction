# ml-traffic-prediction
An end-to-end machine learning pipeline to predict traffic volume using historical traffic, weather, and temporal data; performed feature engineering, model comparison, and evaluation, and saved the best-performing Random Forest model for real-time prediction.



## **Traffic Volume Prediction Using Machine Learning**

---

## **Problem Statement:**

**Traffic Volume Prediction on Roadways**

The aim of this project is to predict **traffic volume on a given roadway** by leveraging **machine learning techniques** and various **environmental, temporal, and weather-related factors**.

Accurate traffic prediction helps in:

* Traffic management and planning
* Supporting intelligent transportation systems

The model uses **historical traffic data**, including **weather conditions, holidays, and time-based features**, to estimate the number of vehicles on the road at a given time.

---

## **High-Level Steps Involved:**

1. **Dataset Loading**
   The traffic dataset was loaded using the pandas library for data analysis and preprocessing.

2. **Handling Missing Values**
   Missing values in the dataset (such as holidays) were identified and treated appropriately to ensure data consistency.

3. **Date-Time Feature Engineering**
   Since machine learning models cannot process raw date-time values, the date and time column was converted into meaningful numerical features such as:

   * Hour
   * Day
   * Month
   * Weekday

4. **Categorical Data Conversion**
   Categorical weather features such as `weather_main` and `weather_description` were converted into numerical format using **one-hot encoding**.

5. **Data Visualization**
   Seaborn was used to visualize the distribution of traffic volume to understand data patterns and trends.

6. **Feature and Target Separation**
   The dataset was split into:

   * Features (independent variables)
   * Target variable (`traffic_volume`)

7. **Train-Test Split**
   The dataset was divided into training and testing sets to evaluate model performance on unseen data.

8. **Model Training**
   Three regression models were trained:

   * Linear Regression
   * Decision Tree Regressor
   * Random Forest Regressor

9. **Model Evaluation**
   Models were evaluated using **Mean Absolute Error (MAE)** to determine prediction accuracy.

10. **Model Selection and Saving**
    The best-performing model (Random Forest Regressor) was saved using **joblib** for future deployment and real-time prediction.


## **Dataset Used:**

Traffic and weather dataset containing:

* Traffic volume
* Weather conditions
* Holiday information
* Date and time details

---

## **Evaluation Metric Used:**

**Mean Absolute Error (MAE)**
Used to measure the average magnitude of prediction errors.

---

## **Final Outcome:**

Among the three models trained, the **Random Forest Regressor** achieved the lowest error and provided the most accurate traffic volume predictions. The trained model was saved successfully and can be used for **real-time traffic prediction applications**

Just tell me 💙
