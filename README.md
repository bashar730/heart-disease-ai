# heart-disease-ai
Heart disease risk prediction system using machine learning and Streamlit.

#

## نظام ذكي للتنبؤ باحتمالية الإصابة بأمراض القلب

مشروع تخرج بحثي وتعليمي يهدف إلى استخدام تقنيات تعلم الآلة لتحليل مجموعة من المؤشرات الصحية وتقديم نتيجة تنبؤية أولية لاحتمالية الانتماء إلى الفئة الإيجابية وفق النموذج.

## Dataset

استخدم المشروع مجموعة بيانات Cleveland التابعة لمستودع UCI Heart Disease، وتضم 303 سجلات و13 خاصية طبية مستخدمة في التنبؤ.

عُولجت القيم المفقودة في الخاصيتين `ca` و`thal` باستخدام المنوال، ثم حُوّل المتغير المستهدف `num` إلى تصنيف ثنائي:

* `0`: عدم وجود مرض في البيانات.
* `1`: وجود مرض في البيانات.

## Machine Learning Models

دُربت وقورنت مجموعة من خوارزميات تعلم الآلة، ثم اختير نموذج Logistic Regression بعد تقييم التوازن بين الأداء والثبات وقابلية التفسير.

يتكون النموذج النهائي من Pipeline تشمل:

1. `StandardScaler`
2. `LogisticRegression`

## Final Model Results

* Accuracy: 86.89%
* Precision: 81.25%
* Recall: 92.86%
* F1-score: 86.67%
* ROC-AUC: 95.02%

## Explainability

يستخدم التطبيق مكتبة SHAP لتوضيح مساهمة الخصائص في رفع مخرج النموذج أو خفضه. ولا تمثل قيم SHAP أسبابًا طبية أو علاقات سببية.

## Repository Structure

```text
data/
├── raw/
└── processed/

notebooks/
├── 01_Data_Understanding.ipynb
├── 02_Data_Preprocessing.ipynb
└── 03_Model_Training_Evaluation.ipynb

models/
└── heart_disease_model.pkl

app/
└── app.py

reports/
└── figures/

requirements.txt
README.md
```

## Project Branches

* `data-eda`: فهم البيانات وتحليلها ومعالجتها.
* `model-evaluation`: تدريب النماذج ومقارنتها وتقييمها.
* `streamlit-docs`: فرع التكامل النهائي وتطوير الواجهة والتفسير والتوثيق والنشر.

جُمعت مكونات المشروع النهائية في فرع `streamlit-docs`، ومنه نُشر التطبيق على Streamlit Community Cloud.

## Running the Application

ثبت المكتبات المطلوبة:

```bash
pip install -r requirements.txt
```

ثم شغّل التطبيق:

```bash
streamlit run app/app.py
```

## Online Application

رابط التطبيق المنشور:

ضع رابط Streamlit Community Cloud هنا

## Team Responsibilities

* العضو الأول: البيانات والتحليل الاستكشافي والمعالجة.
* العضو الثاني: تدريب النماذج وضبطها وتقييمها.
* العضو الثالث: تطوير النظام وتفسير النتائج والتوثيق والنشر.

## Medical Disclaimer

هذا النظام مشروع بحثي وتعليمي لدعم فهم تطبيقات تعلم الآلة، ولا يقدم تشخيصًا طبيًا، ولا يستبدل الطبيب أو الفحوصات الطبية المعتمدة.

