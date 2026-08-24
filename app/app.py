"""
واجهة Streamlit عربية لمشروع BNN للتنبؤ بأمراض القلب.
تجمع الواجهة 13 قيمة، وترتبها مثل التدريب، ثم ترسلها للنموذج المحفوظ.
الناتج نسبة احتمالية تعليمية فقط، وليس تشخيصًا أو بديلًا عن الطبيب.
"""

from pathlib import Path          # لبناء مسار ملف النموذج بطريقة تعمل محليًا وعلى Streamlit Cloud
import joblib                    # لقراءة النموذج المحفوظ بصيغة pkl
import pandas as pd              # لتحويل المدخلات إلى جدول يفهمه النموذج
import streamlit as st           # لبناء عناصر الواجهة وتشغيل التطبيق

# 1) إعداد الصفحة: centered يجعل المحتوى ضيقًا مثل تطبيق جوال على الكمبيوتر والهاتف
st.set_page_config(
    page_title="نظام BNN للتنبؤ بأمراض القلب",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# 2) تنسيق بسيط داخل نفس الملف: RTL للعربية، وإخفاء الشريط الجانبي والخط الخاص به
st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none;}
.block-container {max-width:650px; padding:1rem 1rem 3rem;}
html, body, [class*="st-"] {direction:rtl; text-align:right;}
.title {background:linear-gradient(135deg,#14213d,#27496d); color:white;
        padding:22px; border-radius:20px; text-align:center; margin-bottom:18px;}
.title h1 {font-size:1.65rem; margin:0 0 8px;} .title p {margin:0; color:#dce8f5;}
div[data-testid="stForm"] {background:white; border:1px solid #e4e9f1;
        padding:18px; border-radius:20px; box-shadow:0 8px 24px #14213d12;}
h3 {color:#203864; border-right:4px solid #d94b5a; padding-right:10px;}
div[data-testid="stFormSubmitButton"] button {width:100%; min-height:48px;
        border-radius:12px; font-weight:700;}
.result {padding:22px; border-radius:18px; text-align:center; margin-top:18px;}
.low {background:#ecfdf3; border:1px solid #61c985; color:#17663a;}
.high {background:#fff1f2; border:1px solid #ef7b88; color:#9f2434;}
.percent {font-size:2.5rem; font-weight:800; direction:ltr;}
.note {background:#fff8e8; border:1px solid #efd27c; padding:14px;
       border-radius:14px; margin-top:18px; color:#72520c;}
</style>
""", unsafe_allow_html=True)


# 3) أسماء الخصائص وترجمتها؛ ترتيب القاموس هو ترتيب التدريب ولا يجب تغييره
LABELS = {
    "age": "العمر", "sex": "الجنس", "cp": "نوع ألم الصدر",
    "trestbps": "ضغط الدم", "chol": "الكوليسترول", "fbs": "سكر الصيام",
    "restecg": "تخطيط القلب", "thalach": "أقصى معدل لضربات القلب",
    "exang": "ذبحة المجهود", "oldpeak": "انخفاض ST", "slope": "ميل ST",
    "ca": "عدد الأوعية الرئيسية", "thal": "اختبار الثاليوم",
}
FEATURES = list(LABELS)
# app.py داخل مجلد app، لذلك parents[1] يرجع إلى جذر المشروع ثم يدخل مجلد models
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "heart_disease_model.pkl"


# 4) تحميل النموذج مرة واحدة فقط لتسريع التطبيق عند تكرار الضغط على زر التحليل
@st.cache_resource(show_spinner=False)
def load_model():
    bundle = joblib.load(MODEL_PATH)                    # الحزمة فيها النموذج وأسماء الخصائص
    if list(bundle.get("feature_names", [])) != FEATURES:
        raise ValueError("خصائص النموذج لا تطابق خصائص الواجهة.")
    return bundle["model"]


# دالة مساعدة: تعرض النص العربي للمستخدم، لكنها تعيد الرقم الذي يحتاجه النموذج
def choose(label, options, default):
    codes = list(options)
    return st.selectbox(label, codes, index=codes.index(default), format_func=options.get)
# يحسب مساهمة كل خاصية في نتيجة الانحدار اللوجستي بعد تطبيق StandardScaler
def explain_result(model, data):
    scaled = model.named_steps["scaler"].transform(data)[0]
    weights = model.named_steps["logistic"].coef_[0]
    ranked = sorted(zip(FEATURES, scaled * weights), key=lambda item: abs(item[1]), reverse=True)
    increase = [LABELS[name] for name, effect in ranked if effect > 0][:3]
    decrease = [LABELS[name] for name, effect in ranked if effect < 0][:3]
    return increase, decrease

# 5) تحويل الاختيارات الطبية المكتوبة إلى الرموز الرقمية المستخدمة في بيانات UCI
SEX = {1: "ذكر", 0: "أنثى"}
CP = {1: "ذبحة نموذجية", 2: "ذبحة غير نموذجية", 3: "ألم غير ذبحي", 4: "بدون أعراض"}
YES_NO = {0: "لا", 1: "نعم"}
ECG = {0: "طبيعي", 1: "شذوذ ST-T", 2: "تضخم البطين الأيسر"}
SLOPE = {1: "صاعد", 2: "مسطح", 3: "هابط"}
CA = {0: "0 - صفر", 1: "1 - وعاء", 2: "2 - وعاءان", 3: "3 - ثلاثة أوعية"}
THAL = {3: "طبيعي", 6: "عيب ثابت", 7: "عيب قابل للعكس"}


# 6) رأس الصفحة الذي يوضح اسم النظام ووظيفته
st.markdown("""<div class="title"><h1>🫀 نظام BNN للتنبؤ بأمراض القلب</h1>
<p>أدخل المؤشرات الطبية للحصول على تقدير مبدئي من نموذج تعلم الآلة</p></div>""",
            unsafe_allow_html=True)


# 7) النموذج المرئي: كل المدخلات عمودية وبعرض كامل لتناسب شاشة الجوال
with st.form("heart_form"):
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
    submitted = st.form_submit_button("تحليل البيانات وإظهار النتيجة", type="primary",
                                      use_container_width=True)


# 8) هذا الجزء لا يعمل إلا بعد الضغط على الزر: يجمع القيم ويتنبأ ثم يعرض النتيجة
if submitted:
    try:
        values = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]
        data = pd.DataFrame([values], columns=FEATURES)  # صف واحد مرتب بأسماء الخصائص
        model = load_model()
        probability = float(model.predict_proba(data)[0, 1]) * 100  # احتمال الفئة 1
        prediction = int(model.predict(data)[0])                    # القرار: 0 أو 1
        style = "high" if prediction == 1 else "low"
        text = "احتمال مرتفع وفق النموذج" if prediction == 1 else "احتمال منخفض وفق النموذج"
        icon = "⚠️" if prediction == 1 else "✅"
        st.markdown(f"""<div class="result {style}"><div class="percent">{probability:.1f}%</div>
        <h2>{icon} {text}</h2><p>هذه نتيجة تعليمية تقديرية وليست تشخيصًا طبيًا.</p></div>""",
                    unsafe_allow_html=True)
        st.progress(probability / 100)                  # شريط بصري من صفر إلى واحد
        increase, decrease = explain_result(model, data)
        st.subheader("تفسير النتيجة")
        st.success("عوامل رفعت تقدير النموذج: " + "، ".join(increase))
        st.info("عوامل خفّضت تقدير النموذج: " + "، ".join(decrease))
        st.caption("هذه تأثيرات حسابية داخل النموذج وليست أسبابًا طبية أو علاقة سببية.")
    except Exception as error:
        st.error(f"تعذّر إجراء التنبؤ: {error}")       # يعرض سبب المشكلة بدل توقف الصفحة


# 9) تنبيه ثابت يوضح حدود استخدام مشروع التخرج
st.markdown("""<div class="note"><b>تنبيه طبي:</b> هذا النظام مشروع تعليمي لدعم القرار،
ولا يستبدل الطبيب أو الفحوصات الطبية المعتمدة.</div>""", unsafe_allow_html=True)
