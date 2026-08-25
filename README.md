<div align="center">

<h1>Student Outcome Prediction — ML Classification Model</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-86%25-green?style=for-the-badge"/>
</p>

</div>

<div style="background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%); padding: 18px; border-radius: 12px; border-left: 5px solid #38bdf8; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;">
<p style="color: #cdd6f4; margin: 0; line-height: 1.7; font-size: 15px;">
An end-to-end machine learning system and interactive web application designed to predict student academic success and analyze engagement patterns using the Open University Learning Analytics Dataset (OULAD).
</p>
</div>

---

<h2>📊 Results Summary</h2>

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; border-color: #313244;">
  <thead>
    <tr style="background-color: #181825; color: #89b4fa;">
      <th align="left">Metric / Aspect</th>
      <th align="left">Details</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Model Accuracy</b></td>
      <td>Achieved an overall accuracy of <b>86%</b> on the test dataset.</td>
    </tr>
    <tr>
      <td><b>Performance Balance</b></td>
      <td>Strong classification metrics across both classes (Pass/Fail) with reliable precision and recall scores.</td>
    </tr>
  </tbody>
</table>

---

<h2>📁 Dataset</h2>

<p>
  The dataset used in this project is the multi-table <b>Open University Learning Analytics Dataset (OULAD)</b>. You can access and download the dataset directly from <a href="https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad" target="_blank">Kaggle Dataset Link</a>.
</p>

<p><b>Place all CSV files in the root project folder before running!</b></p>

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; border-color: #313244;">
  <thead>
    <tr style="background-color: #181825; color: #89b4fa;">
      <th align="left">File</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>studentInfo.csv</code></td>
      <td>Demographics & final outcomes</td>
    </tr>
    <tr>
      <td><code>studentVle.csv</code></td>
      <td>Student activity logs (VLE interactions)</td>
    </tr>
    <tr>
      <td><code>studentAssessment.csv</code></td>
      <td>Assessment scores</td>
    </tr>
    <tr>
      <td><code>studentRegistration.csv</code></td>
      <td>Registration & withdrawal info</td>
    </tr>
    <tr>
      <td><code>courses.csv</code></td>
      <td>Course-level info</td>
    </tr>
    <tr>
      <td><code>assessments.csv</code></td>
      <td>Assessment metadata</td>
    </tr>
    <tr>
      <td><code>vle.csv</code></td>
      <td>Virtual learning environment tools</td>
    </tr>
  </tbody>
</table>

---

<h2>⚙️ Project Pipeline</h2>

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; border-color: #313244;">
  <thead>
    <tr style="background-color: #181825; color: #89b4fa;">
      <th align="left">Pipeline Stage</th>
      <th align="left">Actions & Implementation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1. Data Loading & Exploration</b></td>
      <td>Loaded raw CSV datasets including student info, VLE clicks, and assessments; explored structural relationships and identified missing values.</td>
    </tr>
    <tr>
      <td><b>2. Data Cleaning</b></td>
      <td>Cleaned and preprocessed raw datasets to handle missing records, standardize formats, and map unregistration dates for proper binary tracking.</td>
    </tr>
    <tr>
      <td><b>3. Feature Engineering</b></td>
      <td>Calculated key behavioral features per student: <code>avg_score</code>, <code>num_assessments</code>, <code>total_clicks</code>, <code>active_days</code>, <code>avg_clicks_per_day</code>, and <code>withdrew</code> status.</td>
    </tr>
    <tr>
      <td><b>4. Model Training</b></td>
      <td>Built a binary classification target, encoded categorical features using <code>pd.get_dummies</code>, and trained a <b>Random Forest Classifier</b> (80/20 train-test split).</td>
    </tr>
    <tr>
      <td><b>5. Deployment</b></td>
      <td>Exported the trained model using <code>joblib</code> (<code>student_model.pkl</code>) and generated historical processed data for real-time comparative percentile ranking.</td>
    </tr>
  </tbody>
</table>

---

<h2>🚀 How to Run</h2>

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; border-color: #313244;">
  <thead>
    <tr style="background-color: #181825; color: #89b4fa;">
      <th align="left">Step</th>
      <th align="left">Command / Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Step 1</b></td>
      <td>Install dependencies:<br><code>pip install pandas numpy scikit-learn streamlit joblib</code></td>
    </tr>
    <tr>
      <td><b>Step 2</b></td>
      <td>Run the ML pipeline script to process data, train the model, and output the model files:<br><code>python student.py</code></td>
    </tr>
    <tr>
      <td><b>Step 3</b></td>
      <td>Launch the Streamlit web application locally:<br><code>streamlit run apps.py</code></td>
    </tr>
  </tbody>
</table>

---

<h2>💻 Streamlit App Features</h2>

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; border-color: #313244;">
  <thead>
    <tr style="background-color: #181825; color: #89b4fa;">
      <th align="left">Feature</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Real-time Predictions</b></td>
      <td>Input student metrics via an interactive sidebar to instantly predict success probability and risk levels.</td>
    </tr>
    <tr>
      <td><b>Comparative Analytics</b></td>
      <td>Computes student performance and engagement scores against historical data percentiles.</td>
    </tr>
    <tr>
      <td><b>Probability Distributions</b></td>
      <td>Visualizes success vs. failure probabilities dynamically through interactive charts.</td>
    </tr>
    <tr>
      <td><b>Personalized Recommendations</b></td>
      <td>Generates actionable academic insights and interventions based on student behavior.</td>
    </tr>
  </tbody>
</table>

---

<h2>📦 Tech Stack</h2>

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; border-color: #313244;">
  <thead>
    <tr style="background-color: #181825; color: #89b4fa;">
      <th align="left">Category</th>
      <th align="left">Technologies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Programming Language</b></td>
      <td>Python</td>
    </tr>
    <tr>
      <td><b>Machine Learning</b></td>
      <td>Scikit-learn, Random Forest</td>
    </tr>
    <tr>
      <td><b>Data Processing</b></td>
      <td>Pandas, NumPy</td>
    </tr>
    <tr>
      <td><b>Web Framework</b></td>
      <td>Streamlit</td>
    </tr>
    <tr>
      <td><b>Model Persistence</b></td>
      <td>Joblib</td>
    </tr>
    <tr>
      <td><b>Visualization</b></td>
      <td>Streamlit native charts & UI components</td>
    </tr>
  </tbody>
</table>

---

<h2>📁 Project Structure</h2>

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; border-color: #313244;">
  <thead>
    <tr style="background-color: #181825; color: #89b4fa;">
      <th align="left">File Name</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>student.py</code></td>
      <td>Data preprocessing, feature engineering, and model training script</td>
    </tr>
    <tr>
      <td><code>apps.py</code></td>
      <td>Interactive Streamlit web application interface</td>
    </tr>
    <tr>
      <td><code>studentInfo.csv</code></td>
      <td>Student demographic dataset</td>
    </tr>
    <tr>
      <td><code>studentVle.csv</code></td>
      <td>Student Virtual Learning Environment interactions</td>
    </tr>
    <tr>
      <td><code>studentAssessment.csv</code></td>
      <td>Student assessment scores</td>
    </tr>
    <tr>
      <td><code>studentRegistration.csv</code></td>
      <td>Registration and unregistration details</td>
    </tr>
    <tr>
      <td><code>student_model.pkl</code></td>
      <td>Saved trained Random Forest model</td>
    </tr>
    <tr>
      <td><code>decoded_student_data.csv</code></td>
      <td>Processed historical dataset for comparisons</td>
    </tr>
  </tbody>
</table>

---

<h2>👤 Author</h2>
<p><b>Khaled Tayeb</b><br>
Computer Science — King Abdulaziz University | AI & Data Analyst</p>
