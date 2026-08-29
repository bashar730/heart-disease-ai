
from pathlib import Path
import joblib
import pandas as pd
import shap
import streamlit as st

# إعداد صفحة التطبيق
st.set_page_config(
    page_title="نظام BNN لأمراض القلب",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# تنسيق الواجهة ودعم اللغة العربية
st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
    display: none;
}
.block-container {
    max-width: 720px;
    padding: 1rem 1rem 3rem;
}
html, body, [class*="st-"] {
    direction: rtl;
    text-align: right;
}
.title-card {
    background: linear-gradient(135deg, #14213d, #27496d);
    color: white;
    padding: 24px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 18px;
}
.title-card h1 {
    font-size: 1.7rem;
    margin: 0 0 8px;
}
.title-card p {
    margin: 0;
    color: #dce8f5;
}
.result {
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    margin-top: 18px;
}
.low {
    background: #ecfdf3;
    border: 1px solid #61c985;
    color: #17663a;
}
.high {
    background: #fff1f2;
    border: 1px solid #ef7b88;
    color: #9f2434;
}
.percent {
    font-size: 2.6rem;
    font-weight: 800;
    direction: ltr;
}
.note {
    background: #fff8e8;
    border: 1px solid #efd27c;
    padding: 14px;
    border-radius: 14px;
    margin-top: 18px;
    color: #72520c;
}
</style>
""", unsafe_allow_html=True)

# أسماء الخصائص الطبية وترجمتها
LABELS = {
    "age": "العمر",
    "sex": "الجنس",
    "cp": "نوع ألم الصدر",
    "trestbps": "ضغط الدم أثناء الراحة",
    "chol": "الكوليسترول",
    "fbs": "سكر الصيام",
    "restecg": "تخطيط القلب",
    "thalach": "أقصى معدل لضربات القلب",
    "exang": "ذبحة المجهود",
    "oldpeak": "انخفاض مقطع ST",
    "slope": "ميل مقطع ST",
    "ca": "عدد الأوعية الرئيسية",
    "thal": "اختبار الثاليوم"
}

FEATURES = list(LABELS)
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "heart_disease_model.pkl"
)

# تحميل النموذج وإنشاء مفسر SHAP مرة واحدة
@st.cache_resource(show_spinner=False)
def load_resources():
    bundle = joblib.load(MODEL_PATH)

    if list(bundle.get("feature_names", [])) != FEATURES:
        raise ValueError("خصائص النموذج لا تطابق خصائص الواجهة.")

    model = bundle["model"]
    scaler = model.named_steps["scaler"]
    logistic = model.named_steps["logistic"]

    background_raw = pd.DataFrame(
        [scaler.mean_],
        columns=FEATURES
    )
    background_scaled = scaler.transform(background_raw)

    masker = shap.maskers.Independent(
        background_scaled,
        max_samples=1
    )
    explainer = shap.LinearExplainer(logistic, masker)

    return model, explainer

# عرض الاختيارات العربية وإرجاع الرمز الرقمي للنموذج
def choose(label, options, default, key):
    codes = list(options)
    return st.selectbox(
        label,
        codes,
        index=codes.index(default),
        format_func=options.get,
        key=key
    )

# حساب أهم العوامل المؤثرة باستخدام SHAP
def explain_result(model, explainer, data):
    scaled_data = model.named_steps["scaler"].transform(data)
    values = explainer(scaled_data).values[0]

    ranked = sorted(
        zip(FEATURES, values),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    increase = [(name, value) for name, value in ranked if value > 0][:3]
    decrease = [(name, value) for name, value in ranked if value < 0][:3]
    return increase, decrease

# إعادة الحقول إلى القيم الافتراضية
def clear_form():
    for key in FEATURES:
        st.session_state.pop(key, None)

SEX = {1: "ذكر", 0: "أنثى"}
CP = {
    1: "ذبحة نموذجية",
    2: "ذبحة غير نموذجية",
    3: "ألم غير ذبحي",
    4: "بدون أعراض"
}
YES_NO = {0: "لا", 1: "نعم"}
ECG = {0: "طبيعي", 1: "شذوذ ST-T", 2: "تضخم البطين الأيسر"}
SLOPE = {1: "صاعد", 2: "مسطح", 3: "هابط"}
CA = {0: "صفر", 1: "وعاء واحد", 2: "وعاءان", 3: "ثلاثة أوعية"}
THAL = {3: "طبيعي", 6: "عيب ثابت", 7: "عيب قابل للعكس"}

# عنوان التطبيق
st.markdown("""
<div class="title-card">
<h1>🫀 نظام BNN للتنبؤ بأمراض القلب</h1>
<p>أدخل المؤشرات الطبية للحصول على تقدير تعليمي وتفسير باستخدام SHAP</p>
</div>
""", unsafe_allow_html=True)

# نموذج إدخال الخصائص الثلاث عشرة
with st.form("heart_form"):
    st.subheader("المعلومات الأساسية")
    age = st.number_input("العمر (سنة)", 18, 100, 50, key="age")
    sex = choose("الجنس", SEX, 1, "sex")
    cp = choose("نوع ألم الصدر", CP, 1, "cp")

    st.subheader("القياسات والفحوصات")
    trestbps = st.number_input(
        "ضغط الدم أثناء الراحة (mmHg)", 70, 250, 120, key="trestbps"
    )
    chol = st.number_input(
        "الكوليسترول (mg/dL)", 80, 700, 200, key="chol"
    )
    fbs = choose("هل سكر الصيام أكبر من 120؟", YES_NO, 0, "fbs")
    restecg = choose("نتيجة تخطيط القلب", ECG, 0, "restecg")

    st.subheader("اختبار الجهد والقلب")
    thalach = st.number_input(
        "أقصى معدل لضربات القلب", 40, 230, 150, key="thalach"
    )
    exang = choose("هل ظهرت ذبحة بسبب المجهود؟", YES_NO, 0, "exang")
    oldpeak = st.number_input(
        "انخفاض مقطع ST", 0.0, 10.0, 1.0, 0.1, key="oldpeak"
    )
    slope = choose("ميل مقطع ST", SLOPE, 1, "slope")
    ca = choose("عدد الأوعية الرئيسية", CA, 0, "ca")
    thal = choose("نتيجة اختبار الثاليوم", THAL, 3, "thal")

    first_button, second_button = st.columns(2)

    with first_button:
        submitted = st.form_submit_button(
            "تحليل البيانات",
            type="primary",
            use_container_width=True
        )

    with second_button:
        st.form_submit_button(
            "مسح الحقول",
            on_click=clear_form,
            use_container_width=True
        )

# تنفيذ التنبؤ والتفسير
if submitted:
    try:
        values = [
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]

        data = pd.DataFrame([values], columns=FEATURES)
        model, explainer = load_resources()

        probability = float(model.predict_proba(data)[0, 1])
        prediction = int(model.predict(data)[0])

        css_class = "high" if prediction == 1 else "low"
        status = (
            "احتمال مرتفع وفق النموذج"
            if prediction == 1
            else "احتمال منخفض وفق النموذج"
        )
        icon = "⚠️" if prediction == 1 else "✅"

        st.markdown(f"""
        <div class="result {css_class}">
        <div class="percent">{probability * 100:.1f}%</div>
        <h2>{icon} {status}</h2>
        <p>هذه نتيجة تقديرية تعليمية وليست تشخيصًا طبيًا.</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(probability)

        increase, decrease = explain_result(
            model, explainer, data
        )

        increase_text = "، ".join(
            LABELS[name] for name, value in increase
        )
        decrease_text = "، ".join(
            LABELS[name] for name, value in decrease
        )

        st.subheader("تفسير النتيجة باستخدام SHAP")
        st.success("عوامل رفعت تقدير النموذج: " + increase_text)
        st.info("عوامل خفّضت تقدير النموذج: " + decrease_text)

        st.caption(
            "قِيَم SHAP توضّح تأثير الخصائص داخل النموذج، "
            "ولا تمثل أسبابًا طبية أو علاقة سببية."
        )

    except Exception as error:
        st.error(f"تعذّر إجراء التنبؤ: {error}")

# التنبيه الطبي الثابت
st.markdown("""
<div class="note">
<b>تنبيه طبي:</b>
هذا النظام مشروع تعليمي لدعم القرار، ولا يستبدل الطبيب
أو الفحوصات الطبية المعتمدة.
</div>
""", unsafe_allow_html=True)
