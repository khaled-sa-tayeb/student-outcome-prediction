<div align="center">

<h1>🎓 Student Outcome Prediction — ML Classification Model & Power BI Dashboard</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black"/>
  <img src="https://img.shields.io/badge/Accuracy-86%25-green?style=for-the-badge"/>
</p>

</div>

<div style="background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%); padding: 18px; border-radius: 12px; border-left: 5px solid #38bdf8; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;">
<p style="color: #cdd6f4; margin: 0; line-height: 1.7; font-size: 15px;">
An end-to-end machine learning system and interactive web application designed to predict student academic success and analyze engagement patterns using the Open University Learning Analytics Dataset (OULAD).
</p>
</div>

---

<h2>📊 Results</h2>
<ul>
  <li><b>Model Accuracy:</b> Achieved an overall accuracy of <b>86%</b> on the test dataset[cite: 2].</li>
  <li><b>Balanced Performance:</b> Strong classification metrics across both classes (Pass/Fail) with reliable precision and recall scores[cite: 2].</li>
</ul>

---

<h2>📸 Power BI Dashboard</h2>
<ul>
  <li>Integrated interactive Power BI dashboards to track student performance distributions, VLE engagement metrics, and risk factors.</li>
</ul>

---

<h2>📁 Dataset</h2>
<ul>
  <li>Utilizes the multi-table <b>Open University Learning Analytics Dataset (OULAD)</b>, which includes student demographics, VLE interactions (<code>studentVle.csv</code>), assessment results (<code>studentAssessment.csv</code>), and registration details (<code>studentRegistration.csv</code>)[cite: 2].</li>
</ul>

---

<h2>⚙️ Project Pipeline</h2>

<h3>1. Data Loading & Exploration</h3>
<ul>
  <li>Loaded raw CSV datasets including student info, VLE clicks, and assessments[cite: 2].</li>
  <li>Explored structural relationships and identified missing values across multiple tables[cite: 2].</li>
</ul>

<h3>2. Data Cleaning</h3>
<ul>
  <li>Cleaned and preprocessed raw datasets to handle missing records and standardize formats.</li>
  <li>Mapped unregistration dates to build proper binary tracking metrics[cite: 2].</li>
</ul>

<h3>3. Feature Engineering</h3>
<ul>
  <li>Calculated key academic and behavioral features per student:
    <ul>
      <li>Average assessment scores (<code>avg_score</code>)[cite: 2]</li>
      <li>Total submitted assessments (<code>num_assessments</code>)[cite: 2]</li>
      <li>Total VLE clicks (<code>total_clicks</code>)[cite: 2]</li>
      <li>Active study days (<code>active_days</code>)[cite: 2]</li>
      <li>Average daily clicks (<code>avg_clicks_per_day</code>)[cite: 2]</li>
      <li>Withdrawal status tracking (<code>withdrew</code>)[cite: 2]</li>
    </ul>
  </li>
</ul>

<h3>4. Model Training</h3>
<ul>
  <li>Built a binary classification target (<code>target</code>) separating successful completions (Pass/Distinction) from non-completion[cite: 2].</li>
  <li>Encoded categorical features using Pandas one-hot encoding (<code>pd.get_dummies</code>)[cite: 2].</li>
  <li>Trained a <b>Random Forest Classifier</b> (<code>RandomForestClassifier</code>) using Scikit-learn, split into 80% training and 20% testing sets[cite: 2].</li>
</ul>

<h3>5. Deployment</h3>
<ul>
  <li>Exported the trained model using <code>joblib</code> (<code>student_model.pkl</code>)[cite: 2].</li>
  <li>Generated a processed historical dataset (<code>decoded_student_data.csv</code>) for real-time comparative percentile ranking[cite: 1, 2].</li>
</ul>

---

<h2>🚀 How to Run</h2>

<h3>Step 1 — Install dependencies</h3>
<pre><code>pip install pandas numpy scikit-learn streamlit joblib</code></pre>

<h3>Step 2 — Run the ML pipeline</h3>
<p>Execute the training script to process data, train the model, generate <code>.pkl</code>, and output <code>decoded_student_data.csv</code>:</p>
<pre><code>python student.py</code></pre>

<h3>Step 3 — Launch the Streamlit app</h3>
<p>Run the web application interface locally:</p>
<pre><code>streamlit run apps.py</code></pre>

---

<h2>💻 Streamlit App Features</h2>
<ul>
  <li><b>Real-time Predictions:</b> Input student metrics via an interactive sidebar to instantly predict success probability and risk levels[cite: 1].</li>
  <li><b>Comparative Analytics:</b> Computes student performance and engagement scores against historical data percentiles[cite: 1].</li>
  <li><b>Probability Distributions:</b> Visualizes success vs. failure probabilities dynamically.</li>
  <li><b>Personalized Recommendations:</b> Generates actionable academic insights and interventions based on student behavior.</li>
</ul>

---

<h2>📦 Tech Stack</h2>
<ul>
  <li><b>Programming Language:</b> Python[cite: 1, 2]</li>
  <li><b>Machine Learning:</b> Scikit-learn, Random Forest[cite: 1, 2]</li>
  <li><b>Data Processing:</b> Pandas, NumPy[cite: 1, 2]</li>
  <li><b>Web Framework:</b> Streamlit[cite: 1]</li>
  <li><b>Model Persistence:</b> Joblib[cite: 1, 2]</li>
  <li><b>Visualization:</b> Power BI & Streamlit native charts</li>
</ul>

---

<h2>📁 Project Structure</h2>
<pre><code>├── student.py              # Data preprocessing, feature engineering, and model training script
├── apps.py                 # Interactive Streamlit web application interface
├── studentInfo.csv         # Student demographic dataset
├── studentVle.csv          # Student Virtual Learning Environment interactions
├── studentAssessment.csv   # Student assessment scores
├── studentRegistration.csv # Registration and unregistration details
├── student_model.pkl       # Saved trained Random Forest model
└── decoded_student_data.csv# Processed historical dataset for comparisons</code></pre>

---

<h2>👤 Author</h2>
<p><b>Khaled Tayeb</b><br>
Computer Science Graduate | Data Analyst & AI Enthusiast[cite: 1]</p>
