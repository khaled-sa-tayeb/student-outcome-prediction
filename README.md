<div align="center">

<h1>Student Outcome Prediction — Early-Warning System</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-71%25%20→%2086%25-green?style=for-the-badge"/>
</p>

<p><b><a href="LIVE_APP_URL" target="_blank">▶ Try the live app</a></b></p>

</div>

An end-to-end machine learning system that flags students at risk of failing or withdrawing from an online course, **while there is still time to help**. It is trained on the Open University Learning Analytics Dataset (OULAD): 10.6M platform activity records across 26K+ course enrollments.

Instead of one model, the system has **five models, one per course checkpoint** (day 30, 60, 90, 120 and 150). Each model only sees data available by its day, so early checkpoints give an early warning and later ones are more accurate.

---

<h2>📊 Results</h2>

All checkpoints are evaluated on the **same students, held out from training** (no student appears in both training and test data).

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
  <thead>
    <tr>
      <th align="left">Course day</th>
      <th align="left">Accuracy</th>
      <th align="left">F1 (macro)</th>
      <th align="left">At-risk students caught</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Day 30</td><td>71.4%</td><td>0.71</td><td>66%</td></tr>
    <tr><td>Day 60</td><td>76.4%</td><td>0.75</td><td>67%</td></tr>
    <tr><td>Day 90</td><td>79.7%</td><td>0.78</td><td>69%</td></tr>
    <tr><td>Day 120</td><td>82.0%</td><td>0.81</td><td>72%</td></tr>
    <tr><td><b>Day 150</b></td><td><b>85.7%</b></td><td><b>0.84</b></td><td><b>75%</b></td></tr>
  </tbody>
</table>

<p>Courses last 234–269 days, so day 150 still leaves about three months before the final exam.</p>

---

<h2>⚙️ Pipeline</h2>

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
  <thead>
    <tr><th align="left">Stage</th><th align="left">Details</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1. One row per enrollment</b></td>
      <td>Data is joined on student + module + presentation, so a student taking two modules gets two rows, each with that module's own activity.</td>
    </tr>
    <tr>
      <td><b>2. Feature engineering</b></td>
      <td>Per checkpoint, using only data up to that day: average score, assessments submitted, <b>missed deadlines</b>, <b>average days late</b>, total clicks, active days, clicks per active day, <b>clicks in the last 14 days</b>, <b>days since last activity</b>, plus course and demographic information.</td>
    </tr>
    <tr>
      <td><b>3. No data leakage</b></td>
      <td>No withdrawal or unregistration fields are used as features; students who already withdrew by a checkpoint are excluded from that checkpoint; the train/test split is grouped by student.</td>
    </tr>
    <tr>
      <td><b>4. Model</b></td>
      <td><code>HistGradientBoostingClassifier</code>. Hyperparameters and the decision threshold are chosen with 5-fold <code>GroupKFold</code> cross-validation on the training set only.</td>
    </tr>
    <tr>
      <td><b>5. App</b></td>
      <td>Streamlit app with a checkpoint picker, example students, live predictions, the inputs driving each prediction, comparison with classmates and recommended actions.</td>
    </tr>
  </tbody>
</table>

---

<h2>📁 Dataset</h2>

<p>The raw data is not included in this repository because of its size (<code>studentVle.csv</code> alone is 450MB). Download OULAD from the <a href="https://analyse.kmi.open.ac.uk/open-dataset" target="_blank">Open University</a> or <a href="https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad" target="_blank">Kaggle</a> and place the CSV files in the project root to retrain the models.</p>

---

<h2>🚀 How to Run</h2>

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
  <thead>
    <tr><th align="left">Step</th><th align="left">Command</th></tr>
  </thead>
  <tbody>
    <tr><td><b>1. Install</b></td><td><code>pip install -r requirements.txt</code></td></tr>
    <tr><td><b>2. Run the app</b> (trained models are included)</td><td><code>streamlit run app.py</code></td></tr>
    <tr><td><b>Optional: retrain</b> (needs the dataset)</td><td><code>python student.py</code></td></tr>
  </tbody>
</table>

---

<h2>📁 Project Structure</h2>

<table width="100%" border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
  <thead>
    <tr><th align="left">File</th><th align="left">Description</th></tr>
  </thead>
  <tbody>
    <tr><td><code>student.py</code></td><td>Feature engineering and training of the five checkpoint models</td></tr>
    <tr><td><code>features.py</code></td><td>Feature encoding shared by training and the app, so both prepare inputs identically</td></tr>
    <tr><td><code>app.py</code></td><td>Streamlit web app</td></tr>
    <tr><td><code>Model/day_XX/</code></td><td>Trained model, feature columns and test metrics for each checkpoint</td></tr>
    <tr><td><code>Model/summary.json</code></td><td>Test metrics for all checkpoints</td></tr>
    <tr><td><code>decoded_student_data.csv</code></td><td>Feature values per checkpoint, used by the app to compare a student with classmates</td></tr>
  </tbody>
</table>

---

<h2>📦 Tech Stack</h2>

<p>Python · Pandas · NumPy · Scikit-learn · Streamlit · Altair · Joblib</p>

---

<h2>👤 Author</h2>
<p><b>Khaled Tayeb</b><br>
Computer Science — King Abdulaziz University | AI & Data Analyst</p>
