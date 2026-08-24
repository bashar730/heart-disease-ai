from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# 1) إعداد الصفحة: عرض مركزي ضيق يشبه تطبيق الهاتف
st.set_page_config(
    page_title="نظام BNN للتنبؤ بأمراض القلب",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none;}
    .block-container {max-width:650px; padding:1rem 1rem 3rem;}
    html, body, [class*="st-"] {direction:rtl; text-align:right;}
    .title-card {background:linear-gradient(135deg,#14213d,#203864); color:white;
        padding:22px; border-radius:20px; text-align:center; margin-bottom:18px;}
    .title-card h1 {font-size:1.65rem; margin:0 0 8px;}
    .title-card p {margin:0; color:#dbe6f6;}
    div[data-testid="stForm"] {background:white; border:1px solid #e5eaf2;
        padding:18px; border-radius:20px; box-shadow:0 8px 24px #14213d12;}
    h3 {color:#203864; border-right:4px solid #d94b5a; padding-right:10px;}
    div.stButton > button, div[data-testid="stFormSubmitButton"] button {
        width:100%; border-radius:12px; font-weight:700; min-height:48px;}
    .result {padding:22px; border-radius:18px; text-align:center; margin-top:18px;}
    .low {background:#ecfdf3; border:1px solid #61c985; color:#17663a;}
    .high {background:#fff1f2; border:1px solid #ef7b88; color:#9f2434;}
    .percent {font-size:2.5rem; font-weight:800; direction:ltr;}
    .note {background:#fff8e8; border:1px solid #f1d58a; padding:14px;
        border-radius:14px; margin-top:18px; color:#72520c;}
    </style>
    """,
    unsafe_allow_html=True,
)


# 2) تحميل النموذج والتحقق من ترتيب الخصائص
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_disease_model.pkl"


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("feature_names") != FEATURES:
        raise ValueError("ترتيب خصائص النموذج لا يطابق خصائص الواجهة.")
    return bundle["model"]


def choose(label, options, default):
    keys = list(options)
    return st.selectbox(label, keys, index=keys.index(default), format_func=options.get)


SEX = {1: "ذكر", 0: "أنثى"}
CP = {1: "ذبحة نموذجية", 2: "ذبحة غير نموذجية", 3: "ألم غير ذبحي", 4: "بدون أعراض"}
YES_NO = {0: "لا", 1: "نعم"}
ECG = {0: "طبيعي", 1: "شذوذ ST-T", 2: "تضخم البطين الأيسر"}
SLOPE = {1: "صاعد", 2: "مسطح", 3: "هابط"}
CA = {0: "0 - صفر", 1: "1 - وعاء", 2: "2 - وعاءان", 3: "3 - ثلاثة أوعية"}
THAL = {3: "طبيعي", 6: "عيب ثابت", 7: "عيب قابل للعكس"}


# 3) عنوان النظام
st.markdown(
    """
    <div class="title-card">
      <h1>🫀 نظام BNN للتنبؤ بأمراض القلب</h1>
      <p>أدخل المؤشرات الطبية للحصول على تقدير مبدئي من نموذج تعلم الآلة</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# 4) المدخلات الطبية الثلاثة عشر
with st.form("prediction_form"):
    st.subheader("المعلومات الأساسية")
    age = st.number_input("العمر (سنة)", 18, 100, 50)
    sex = choose("الجنس", SEX, 1)
    cp = choose("نوع ألم الصدر", CP, 1)

    st.subheader("القياسات والفحوصات")
    trestbps = st.number_input("ضغط الدم أثناء الراحة (mmHg)", 70, 250, 120)
    chol = st.number_input("الكوليسترول (mg/dL)", 80, 700, 200)
    fbs = choose("هل سكر الصيام أكبر من 120؟", YES_NO, 0)
    restecg = choose("نتيجة تخطيط القلب أثناء الراحة", ECG, 0)

    st.subheader("اختبار الجهد والقلب")
    thalach = st.number_input("أقصى معدل لضربات القلب", 40, 230, 150)
    exang = choose("هل ظهرت ذبحة بسبب المجهود؟", YES_NO, 0)
    oldpeak = st.number_input("انخفاض مقطع ST (Oldpeak)", 0.0, 10.0, 1.0, 0.1)
    slope = choose("ميل مقطع ST", SLOPE, 1)
    ca = choose("عدد الأوعية الرئيسية", CA, 0)
    thal = choose("نتيجة اختبار الثاليوم", THAL, 3)
    submitted = st.form_submit_button("تحليل البيانات وإظهار النتيجة", type="primary")


# 5) تنفيذ التنبؤ وعرض النتيجة
if submitted:
    try:
        values = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]
        data = pd.DataFrame([values], columns=FEATURES)
        model = load_model()
        probability = float(model.predict_proba(data)[0, 1]) * 100
        prediction = int(model.predict(data)[0])
        css_class = "high" if prediction == 1 else "low"
        status = "احتمال مرتفع وفق النموذج" if prediction == 1 else "احتمال منخفض وفق النموذج"
        icon = "⚠️" if prediction == 1 else "✅"

        st.markdown(
            f"""<div class="result {css_class}"><div class="percent">{probability:.1f}%</div>
            <h2>{icon} {status}</h2><p>هذه نتيجة تقديرية تعليمية وليست تشخيصًا طبيًا.</p></div>""",
            unsafe_allow_html=True,
        )
        st.progress(probability / 100)
    except Exception as error:
        st.error(f"تعذّر إجراء التنبؤ: {error}")


st.markdown(
    """<div class="note"><b>تنبيه طبي:</b> النظام مشروع تعليمي لدعم القرار،
    ولا يستبدل الطبيب أو الفحوصات الطبية المعتمدة.</div>""",
    unsafe_allow_html=True,
)
