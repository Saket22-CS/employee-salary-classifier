# 💼 Employee Salary Classification using Machine Learning

This project predicts whether an employee earns **>50K** or **<=50K** per year using demographic and work-related data. The model is built using the UCI Adult dataset and deployed as an interactive Streamlit web application.

## 🖼️ Project Preview

![App Screenshot](screenshot.png)

## 🚀 Project Overview

- 🔍 **Objective**: Predict salary class (`>50K` or `<=50K`) using features such as age, education level, work class, occupation, etc.
- 🧠 **Internship**: Edunet Foundation – AICTE – IBM SkillsBuild (June–July 2025 Batch)
- 📁 **Dataset**: UCI Machine Learning Repository – Adult Income Dataset
- 🌐 **Web App**: Developed using Streamlit for real-time and batch predictions

---

## 📊 Features

- Data preprocessing and outlier handling
- Label encoding and feature scaling
- Trained and compared multiple ML models
- Selected best-performing model (Gradient Boosting)
- Deployed a web app with:
  - Real-time prediction from user inputs
  - Batch prediction via CSV upload
  - Downloadable prediction results

---

## 🔧 Technologies Used

| Category         | Tools / Libraries                       |
|------------------|------------------------------------------|
| Programming      | Python                                   |
| Data Processing  | Pandas, NumPy                            |
| Visualization    | Matplotlib                               |
| Machine Learning | Scikit-learn                             |
| Model Deployment | Streamlit, Joblib                        |
| Dataset          | [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult) |

---

## 🏗️ Project Structure

```

├── app.py                 # Streamlit application
├── best\_model.pkl        # Saved ML pipeline (scaler + model)
├── adult\_3.csv           # Cleaned dataset (not uploaded here due to size)
├── requirements.txt      # Required Python libraries
├── README.md             # Project documentation

````

---

## 🧪 Model Evaluation

Five models were trained and compared:

| Model                | Accuracy (approx.) |
|----------------------|--------------------|
| Logistic Regression  | 0.84               |
| Random Forest        | 0.86               |
| K-Nearest Neighbors  | 0.83               |
| Support Vector Machine | 0.85             |
| **Gradient Boosting** | **0.87** ✅        |

---

## 🖥️ How to Run the App

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/salary-classification-app.git
cd salary-classification-app
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

4. **Web UI Features:**

   * 🎚️ Enter features like age, education level, occupation, etc.
   * 📈 View individual prediction
   * 📂 Upload CSV for batch prediction
   * ⬇️ Download predicted results

---

## 📌 Sample Input Fields

| Feature         | Input Type     |
| --------------- | -------------- |
| Age             | Slider (18–75) |
| Work Class      | Dropdown       |
| Education Level | Slider (1–16)  |
| Marital Status  | Dropdown       |
| Occupation      | Dropdown       |
| Relationship    | Dropdown       |
| Race            | Dropdown       |
| Gender          | Dropdown       |
| Hours per Week  | Slider (1–80)  |
| Native Country  | Dropdown       |

---

## 📂 Batch Prediction Sample Format

Upload a CSV file with the following headers:

```csv
age,workclass,educational-num,marital-status,occupation,relationship,race,gender,hours-per-week,native-country
35,Private,9,Married-civ-spouse,Sales,Husband,White,Male,40,United-States
```

---

## 📃 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
* Edunet Foundation & AICTE Internship (IBM SkillsBuild)

---

## ✨ Author

**Saket Chaudhary**
B.Tech CSE, Veer Bahadur Singh Purvanchal University /n
📧 [saketrishu64821@gmail.com](mailto:saketrishu64821@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/saket-chaudhary22) 
 [GitHub](https://github.com/Saket22-CS)

---

⭐ If you found this useful, give the repo a star!