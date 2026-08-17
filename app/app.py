import streamlit as st
import joblib
import pandas as pd
import os

# ---------------------------------------------------------
# إعداد الصفحة
# ---------------------------------------------------------
st.set_page_config(page_title="نظام التنبؤ بأمراض القلب", page_icon="❤️", layout="centered")

# ---------------------------------------------------------
# تحميل النموذج (مرة واحدة فقط، ثم يُخزَّن مؤقتاً)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # عدّلي المسار إذا كان مكان الملف مختلف داخل الريبو
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "heart_disease_model.pkl")
    if not os.path.exists(model_path):
        model_path = "heart_disease_model.pkl"  # fallback إذا كان بجانب app.py
    data = joblib.load(model_path)
    return data["model"], data["feature_names"]

model, feature_names = load_model()

# ---------------------------------------------------------
# واجهة المستخدم
# ---------------------------------------------------------
st.title("❤️ نظام التنبؤ بأمراض القلب")
st.write("أدخل بيانات المريض للحصول على تنبؤ باحتمالية الإصابة بمرض القلب.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("العمر (Age)", min_value=1, max_value=120, value=50)
    sex = st.selectbox("الجنس (Sex)", options=[("ذكر", 1), ("أنثى", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox(
        "نوع ألم الصدر (Chest Pain Type)",
        options=[
            ("1 - ذبحة نموذجية", 1),
            ("2 - ذبحة غير نموذجية", 2),
            ("3 - ألم غير ذبحي", 3),
            ("4 - بدون أعراض", 4),
        ],
        format_func=lambda x: x[0],
    )[1]
    trestbps = st.number_input("ضغط الدم أثناء الراحة (Resting BP)", min_value=50, max_value=250, value=120)
    fbs = st.selectbox("سكر الدم الصائم > 120 (Fasting Blood Sugar)", options=[("لا", 0), ("نعم", 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox(
        "نتيجة تخطيط القلب أثناء الراحة (Resting ECG)",
        options=[("0 - طبيعي", 0), ("1 - وجود شذوذ في الموجة ST-T", 1), ("2 - تضخم بطين أيسر محتمل", 2)],
        format_func=lambda x: x[0],
    )[1]
    exang = st.selectbox("ذبحة ناتجة عن مجهود (Exercise Angina)", options=[("لا", 0), ("نعم", 1)], format_func=lambda x: x[0])[1]

with col2:
    chol = st.number_input("الكوليسترول (Cholesterol mg/dl)", min_value=100, max_value=600, value=200)
    thalach = st.number_input("أقصى معدل ضربات قلب (Max Heart Rate)", min_value=60, max_value=220, value=150)
    oldpeak = st.number_input("انخفاض ST الناتج عن المجهود (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox(
        "ميل قطعة ST (Slope)",
        options=[("1 - صاعد", 1), ("2 - مسطح", 2), ("3 - هابط", 3)],
        format_func=lambda x: x[0],
    )[1]
    ca = st.selectbox("عدد الأوعية الرئيسية الملونة (CA)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "اختبار الثاليوم (Thal)",
        options=[("3 - طبيعي", 3), ("6 - عيب ثابت", 6), ("7 - عيب قابل للعكس", 7)],
        format_func=lambda x: x[0],
    )[1]

st.divider()

# ---------------------------------------------------------
# التنبؤ
# ---------------------------------------------------------
if st.button("🔍 احصل على التنبؤ", use_container_width=True):
    input_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    # ترتيب الأعمدة بنفس ترتيب التدريب تماماً
    input_df = pd.DataFrame([input_dict])[feature_names]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("النتيجة:")
    if prediction == 1:
        st.error(f"⚠️ احتمالية وجود مرض قلب: {probability*100:.1f}%")
    else:
        st.success(f"✅ احتمالية منخفضة لمرض القلب: {probability*100:.1f}%")

    st.caption("هذا التنبؤ لغرض تعليمي فقط ولا يُعتبر تشخيصاً طبياً.")
