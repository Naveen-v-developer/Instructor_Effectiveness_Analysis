# 🎓 Instructor Effectiveness Analysis — EdTech Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=flat-square&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red?style=flat-square&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **A machine learning pipeline to define, score, and classify instructor effectiveness across an EdTech platform using batch-level learner outcome, engagement, and feedback data.**

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Critical Analysis](#-critical-analysis)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Technologies Used](#-technologies-used)

---

## 🧠 Problem Statement

An EdTech platform runs the same course across multiple batches taught by different instructors. Each instructor may teach:
- Multiple batches simultaneously
- The same course across different time periods
- Different courses across their tenure

The company wants to **objectively evaluate instructor effectiveness** using data from learner outcomes, engagement patterns, and student feedback — and build an ML model to **predict instructor effectiveness tiers**.

> There is no single correct definition of "effectiveness." This project proposes a principled, weighted composite approach.

---

## 📊 Dataset Description

| Property | Value |
|---|---|
| Total Rows | 2,000 |
| Total Instructors | 120 |
| Total Courses | 25 |
| Batches per Instructor | 7 – 31 (avg ≈ 17) |

Each row represents **one course batch** taught by one instructor.

### Column Reference

#### 🪪 Identifiers
| Column | Description |
|---|---|
| `batch_id` | Unique ID for a course batch |
| `instructor_id` | Unique instructor identifier |
| `course_id` | Course identifier |

#### 📝 Learner Outcome Metrics
| Column | Description |
|---|---|
| `completion_rate` | Fraction of learners who completed the course (0–1) |
| `dropout_rate` | Fraction of learners who dropped out (0–1) |
| `avg_score_improvement` | Average improvement from pre- to post-assessment |
| `avg_quiz_score` | Average quiz score for the batch |

#### 📱 Engagement Metrics
| Column | Description |
|---|---|
| `avg_watch_time` | Normalized average video watch time (0–1) |
| `assignment_submission_rate` | Fraction of learners submitting assignments |
| `forum_activity_rate` | Fraction of learners active on discussion forums |

#### ⭐ Feedback Metrics
| Column | Description |
|---|---|
| `avg_feedback_score` | Average learner feedback rating (1–5) |
| `feedback_response_rate` | Fraction of learners who submitted feedback |

---

## 🔬 Methodology

### Step 1 — Data Aggregation
Batch-level rows are grouped by `instructor_id` to produce one row per instructor with aggregated statistics (mean, std, min, max).

### Step 2 — Effectiveness Score Definition

A **weighted composite score** across 3 pillars:

```
Effectiveness Score =
    0.40 × Learner Outcomes Score
  + 0.35 × Engagement Score
  + 0.25 × Feedback Score
  + 0.05 × Consistency Bonus
```

**Pillar Breakdown:**

| Pillar | Weight | Sub-components |
|---|---|---|
| 📝 Learner Outcomes | 40% | Completion (30%) + Dropout-inverse (30%) + Score Improvement (20%) + Quiz Score (20%) |
| 📱 Engagement | 35% | Watch Time (40%) + Assignment Submission (35%) + Forum Activity (25%) |
| ⭐ Feedback | 25% | Feedback Score (70%) + Feedback Response Rate (30%) |
| 🎯 Consistency Bonus | 5% | Reward for stable performance across batches |

> All metrics are **min-max normalized** to [0, 1] before combining.

### Step 3 — Tier Classification

Instructors are divided into 3 tiers using **percentile cutoffs**:

| Tier | Percentile | Label |
|---|---|---|
| 🥇 Tier 1 | Top 34% | Highly Effective |
| 🥈 Tier 2 | Middle 33% | Moderately Effective |
| 🥉 Tier 3 | Bottom 33% | Needs Improvement |

### Step 4 — ML Model

- **Algorithm:** Random Forest Classifier (300 trees, max depth 8)
- **Features:** 16 instructor-level aggregated columns
- **Validation:** 5-Fold Stratified Cross-Validation
- **CV Accuracy: 85.8% ± 6.2%**

---

## 📁 Project Structure

```
instructor-effectiveness/
│
├── 📓 Instructor_Effectiveness_Analysis.ipynb   # Full annotated notebook
├── 📄 README.md                                 # This file
├── 📊 instructor_effectiveness_dataset.csv      # Input dataset (2000 rows)
├── 📋 instructor_leaderboard.csv                # Output: ranked instructor list
├── 🖼️ real_instructor_report.png               # Output: 9-panel visual report
└── requirements.txt                             # Python dependencies
```

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/instructor-effectiveness.git
cd instructor-effectiveness
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Notebook
```bash
jupyter notebook Instructor_Effectiveness_Analysis.ipynb
```

### 4. Update the Data Path
Inside the notebook, update cell **Step 2** with your CSV filename:
```python
DATA_PATH = "instructor_effectiveness_dataset.csv"
```

### 5. Run All Cells
`Kernel → Restart & Run All`

---

### requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
```

---

## 📈 Results

### Tier Distribution (120 Instructors)

| Tier | Count | % |
|---|---|---|
| 🥇 Tier 1 — Highly Effective | 41 | 34% |
| 🥈 Tier 2 — Moderately Effective | 39 | 33% |
| 🥉 Tier 3 — Needs Improvement | 40 | 33% |

### Model Performance

| Metric | Value |
|---|---|
| CV Accuracy (5-Fold) | **85.8% ± 6.2%** |
| Precision (Tier 1) | 0.93 |
| Recall (Tier 1) | 0.90 |
| F1-Score (macro avg) | 0.91 |

### Top 5 Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | `avg_dropout_rate` | 19.7% |
| 2 | `avg_completion_rate` | 17.2% |
| 3 | `avg_feedback_response_rate` | 12.0% |
| 4 | `max_completion_rate` | 9.6% |
| 5 | `avg_score_improvement` | 7.8% |

### 🏆 Sample Leaderboard

| Rank | Instructor | Batches | Completion Rate | Feedback Score | Effectiveness Score | Tier |
|---|---|---|---|---|---|---|
| 1 | I_010 | 13 | 94.1% | 4.55 | 0.981 | 🥇 Tier 1 |
| 2 | I_037 | 15 | 92.4% | 4.53 | 0.945 | 🥇 Tier 1 |
| 3 | I_018 | 14 | 86.7% | 4.64 | 0.910 | 🥇 Tier 1 |
| ... | ... | ... | ... | ... | ... | ... |
| 120 | I_044 | 15 | 33.4% | 3.73 | 0.109 | 🥉 Tier 3 |

---

## 💡 Key Insights

1. **Dropout rate is the most powerful predictor** — more than quiz scores or feedback ratings. A student quietly quitting is the strongest signal of instructor ineffectiveness.

2. **Feedback response rate matters more than feedback score** — *whether* students engage with the rating form (not just *what* they rate) indicates emotional investment in the course.

3. **Watch time alone is misleading** — students can leave videos playing without watching. It works as a signal only in combination with other engagement metrics.

4. **Score improvement > raw quiz score** — a teacher who takes struggling students from 40% to 70% is more effective than one whose already-strong students score 85%. The delta matters.

5. **Tier 1 vs Tier 3 profiles are clearly separable** — completion rates of 88–94% vs 30–45%, and feedback scores of 4.4–4.7 vs 3.7–4.0, leave little ambiguity at the extremes.

---

## ⚠️ Critical Analysis

### Potentially Misleading Variables

| Variable | Why It Can Mislead |
|---|---|
| `avg_quiz_score` | Reflects course difficulty, not just instructor quality |
| `completion_rate` | Mandatory training inflates this regardless of quality |
| `avg_feedback_score` | Students rate likeable teachers highly even without learning |
| `forum_activity_rate` | Depends on course design, not instructor behavior |
| `avg_watch_time` | Videos left playing ≠ actual attention |

### How the Model Could Fail in Production

- **Goodhart's Law** — Instructors game metrics once they know they're being scored (e.g., making courses easier to boost completion)
- **No student quality control** — An instructor with difficult student cohorts is unfairly penalized vs one with motivated learners
- **Tier boundary brittleness** — Instructors near cutoff lines face binary consequences for negligible score differences
- **Distribution shift** — Model degrades when platform features change (new video format, feedback UI redesign, etc.)
- **Feedback loop bias** — Poor-performing instructors assigned worse batches → scores drop further → unfair cycle

---

## 🚧 Limitations

- No causal inference — correlations found, not causes
- No control for course difficulty or student demographics
- Feedback data only from students who *completed* (survivorship bias)
- Instructor background (experience, qualifications) not included
- Single point-in-time analysis — no trajectory or improvement tracking

---

## 🔮 Future Work

- [ ] Add **student-level data** for value-added modeling (control for who the students are)
- [ ] Include **course difficulty ratings** as a covariate
- [ ] Build **time-series model** to track instructor improvement over semesters
- [ ] Add **SHAP explainability** to show per-instructor score breakdowns
- [ ] Introduce **cohort size weighting** — larger batches should count more
- [ ] Conduct **A/B testing** with same course taught by different instructors to isolate causal effects
- [ ] Build an **instructor dashboard** showing their personal metric trends

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| `Python 3.8+` | Core language |
| `Pandas` | Data manipulation and aggregation |
| `NumPy` | Numerical operations |
| `Scikit-learn` | Random Forest, cross-validation, metrics |
| `Matplotlib` | Visualizations |
| `Seaborn` | Statistical plots and heatmaps |
| `Jupyter Notebook` | Interactive analysis environment |

---

## 🙋 Author

**Your Name**
- 🔗 GitHub: [@Naveen](https://github.com/Naveen-v-developer)

---

> 💬 *"The goal of this model is not to judge instructors — it is to start conversations that help them grow."*
