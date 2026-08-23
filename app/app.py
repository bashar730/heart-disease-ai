"""Arabic Streamlit interface for the BNN heart-disease risk model.

The app collects the exact 13 features used during training, preserves their
order, and sends them to the saved scikit-learn Pipeline. It is an educational
decision-support prototype and must not be presented as a medical diagnosis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Application paths and model contract
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.pkl"
STYLE_PATH = APP_DIR / "styles.css"

EXPECTED_FEATURES = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)

FEATURE_LABELS = {
    "age": "العمر",
    "sex": "الجنس",
    "cp": "نوع ألم الصدر",
    "trestbps": "ضغط الدم أثناء الراحة",
    "chol": "الكوليسترول",
    "fbs": "سكر الدم الصائم",
    "restecg": "تخطيط القلب أثناء الراحة",
    "thalach": "أقصى معدل لضربات القلب",
    "exang": "ذبحة ناتجة عن المجهود",
    "oldpeak": "انخفاض مقطع ST",
    "slope": "ميل مقطع ST",
    "ca": "عدد الأوعية الرئيسية",
    "thal": "نتيجة اختبار الثاليوم",
}

SEX_OPTIONS = {1: "ذكر", 0: "أنثى"}
CHEST_PAIN_OPTIONS = {
    1: "ذبحة صدرية نموذجية",
    2: "ذبحة صدرية غير نموذجية",
    3: "ألم غير ذبحي",
    4: "بدون أعراض",
}
YES_NO_OPTIONS = {0: "لا", 1: "نعم"}
REST_ECG_OPTIONS = {
    0: "طبيعي",
    1: "شذوذ في موجة ST-T",
    2: "احتمال تضخم البطين الأيسر",
}
SLOPE_OPTIONS = {1: "صاعد", 2: "مسطح", 3: "هابط"}
CA_OPTIONS = {
    0: "0 - صفر",
    1: "1 - وعاء واحد",
    2: "2 - وعاءان",
    3: "3 - ثلاثة أوعية",
}
THAL_OPTIONS = {3: "طبيعي", 6: "عيب ثابت", 7: "عيب قابل للعكس"}

# Observed min/max values in the cleaned 303-row training dataset. The input
# widgets allow a wider plausible range, but the result warns about extrapolation.
TRAINING_RANGES = {
    "age": (29.0, 77.0),
    "trestbps": (94.0, 200.0),
    "chol": (126.0, 564.0),
    "thalach": (71.0, 202.0),
    "oldpeak": (0.0, 6.2),
}


# ---------------------------------------------------------------------------
# Page setup and model operations
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="نظام BNN للتنبؤ بأمراض القلب",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_styles() -> None:
    """Load the local stylesheet without depending on an external CDN."""

    if STYLE_PATH.exists():
        st.markdown(
            f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any, tuple[str, ...]]:
    """Load and validate the saved model bundle once per Streamlit process."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict):
        raise TypeError("The model file must contain a dictionary bundle.")

    model = bundle.get("model")
    feature_names = tuple(bundle.get("feature_names", ()))

    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("The saved bundle does not contain a valid classifier.")
    if feature_names != EXPECTED_FEATURES:
        raise ValueError(
            "Model features do not match the 13 features expected by the app."
        )

    return model, feature_names


def make_prediction(
    model: Any, feature_names: tuple[str, ...], values: dict[str, float | int]
) -> dict[str, Any]:
    """Return the class, class-1 probability, and extrapolation warnings."""

    input_frame = pd.DataFrame(
        [{feature: values[feature] for feature in feature_names}],
        columns=feature_names,
    )

    classes = list(getattr(model, "classes_", (0, 1)))
    if 1 not in classes:
        raise ValueError("The classifier does not expose the positive class (1).")

    positive_index = classes.index(1)
    probability = float(model.predict_proba(input_frame)[0][positive_index])
    prediction = int(model.predict(input_frame)[0])

    if not 0.0 <= probability <= 1.0:
        raise ValueError("The model returned an invalid probability.")

    outside_range = []
    for feature, (minimum, maximum) in TRAINING_RANGES.items():
        value = float(values[feature])
        if value < minimum or value > maximum:
            outside_range.append(FEATURE_LABELS[feature])

    return {
        "prediction": prediction,
        "probability": probability,
        "outside_range": outside_range,
    }


# ---------------------------------------------------------------------------
# Reusable UI helpers
# ---------------------------------------------------------------------------


def select_value(
    label: str, options: dict[int, str], default: int, help_text: str
) -> int:
    """Render a coded medical select box with a readable Arabic label."""

    codes = list(options)
    return int(
        st.selectbox(
            label,
            options=codes,
            index=codes.index(default),
            format_func=lambda code: options[int(code)],
            help=help_text,
        )
    )


def display_value(feature: str, value: float) -> str:
    """Convert an encoded model value into a readable summary value."""

    mappings = {
        "sex": SEX_OPTIONS,
        "cp": CHEST_PAIN_OPTIONS,
        "fbs": YES_NO_OPTIONS,
        "restecg": REST_ECG_OPTIONS,
        "exang": YES_NO_OPTIONS,
        "slope": SLOPE_OPTIONS,
        "ca": CA_OPTIONS,
        "thal": THAL_OPTIONS,
    }
    if feature in mappings:
        return mappings[feature][int(value)]

    units = {
        "age": " سنة",
        "trestbps": " mmHg",
        "chol": " mg/dL",
        "thalach": " نبضة/دقيقة",
    }
    if feature == "oldpeak":
        return f"{float(value):.1f}"
    return f"{int(value)}{units.get(feature, '')}"


def build_report(result: dict[str, Any], values: dict[str, float | int]) -> bytes:
    """Create a UTF-8 text report that opens correctly on Windows."""

    probability = result["probability"] * 100
    risk_label = (
        "احتمال مرتفع وفق النموذج"
        if result["prediction"] == 1
        else "احتمال منخفض وفق النموذج"
    )

    lines = [
        "نظام BNN للتنبؤ بأمراض القلب",
        "=" * 36,
        f"النتيجة: {risk_label}",
        f"الاحتمال التقديري: {probability:.1f}%",
        "",
        "البيانات المدخلة:",
    ]
    for feature in EXPECTED_FEATURES:
        lines.append(
            f"- {FEATURE_LABELS[feature]}: {display_value(feature, values[feature])}"
        )

    lines.extend(
        [
            "",
            "تنبيه: هذه النتيجة تعليمية وبحثية، ولا تُعد تشخيصًا طبيًا أو بديلًا عن الطبيب.",
        ]
    )
    return "\n".join(lines).encode("utf-8-sig")


def render_header() -> None:
    st.markdown(
        """
        <section class="hero-card">
          <div class="brand-row">
            <div class="brand-mark" aria-hidden="true">🫀</div>
            <div>
              <span class="brand-kicker">BNN • مشروع تخرج في تقنية المعلومات</span>
              <h1>نظام ذكي لتقدير احتمالية الإصابة بأمراض القلب</h1>
              <p>
                أدخل المؤشرات الصحية المطلوبة، وسيحللها نموذج تعلم الآلة ليعرض
                تقديرًا احتماليًا أوليًا بصورة واضحة وسريعة.
              </p>
            </div>
          </div>
          <div class="hero-pills">
            <span>13 مؤشرًا صحيًا</span>
            <span>نموذج Logistic Regression</span>
            <span>للبحث والتعليم</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🫀 نظام BNN")
        st.caption("دعم قرار تعليمي قائم على تعلم الآلة")
        st.markdown("---")
        st.markdown("### طريقة الاستخدام")
        st.markdown(
            "1. أدخل المعلومات الأساسية.\n"
            "2. أضف القياسات ونتائج الفحوصات.\n"
            "3. اضغط **تحليل البيانات** وشاهد النتيجة."
        )
        st.markdown("---")
        st.markdown("### عن النموذج")
        st.markdown(
            "- **الخوارزمية:** Logistic Regression\n"
            "- **عدد المدخلات:** 13\n"
            "- **البيانات:** 303 سجلات منظفة\n"
            "- **المعالجة:** StandardScaler داخل Pipeline"
        )
        st.info("لا يطلب النظام اسم المريض أو رقمه، ولا ينشئ سجلًا طبيًا دائمًا.")


def render_form() -> tuple[bool, dict[str, float | int]]:
    """Render the 13-feature form and return its submission state and values."""

    st.markdown(
        """
        <div class="section-title">
          <div class="section-icon">01</div>
          <div>
            <h2>بيانات التقييم</h2>
            <p>تنقّل بين الأقسام الثلاثة وأدخل القيم كما تظهر في التقرير الطبي.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("heart_risk_form", clear_on_submit=False):
        basic_tab, measures_tab, tests_tab = st.tabs(
            ["المعلومات الأساسية", "القياسات", "اختبارات القلب"]
        )

        with basic_tab:
            st.caption("معلومات المريض وطبيعة ألم الصدر")
            col_a, col_b = st.columns(2, gap="medium")
            with col_a:
                age = int(
                    st.number_input(
                        "العمر",
                        min_value=18,
                        max_value=100,
                        value=50,
                        step=1,
                        help="العمر بالسنوات. نطاق العمر في بيانات التدريب كان من 29 إلى 77 سنة.",
                    )
                )
                sex = select_value(
                    "الجنس",
                    SEX_OPTIONS,
                    1,
                    "القيمة المسجلة في مجموعة البيانات الطبية.",
                )
            with col_b:
                cp = select_value(
                    "نوع ألم الصدر",
                    CHEST_PAIN_OPTIONS,
                    1,
                    "اختر التصنيف الموجود في تقرير الحالة، وليس تشخيصًا ذاتيًا.",
                )

        with measures_tab:
            st.caption("قياسات الراحة والفحوصات المخبرية الأساسية")
            col_a, col_b = st.columns(2, gap="medium")
            with col_a:
                trestbps = int(
                    st.number_input(
                        "ضغط الدم أثناء الراحة (mmHg)",
                        min_value=70,
                        max_value=250,
                        value=120,
                        step=1,
                        help="ضغط الدم الانقباضي المقاس أثناء الراحة.",
                    )
                )
                chol = int(
                    st.number_input(
                        "الكوليسترول (mg/dL)",
                        min_value=80,
                        max_value=700,
                        value=200,
                        step=1,
                        help="قيمة الكوليسترول الكلي في الدم.",
                    )
                )
            with col_b:
                fbs = select_value(
                    "هل سكر الدم الصائم أكبر من 120 mg/dL؟",
                    YES_NO_OPTIONS,
                    0,
                    "اختر نعم فقط إذا كانت نتيجة سكر الصيام أعلى من 120 mg/dL.",
                )
                restecg = select_value(
                    "نتيجة تخطيط القلب أثناء الراحة",
                    REST_ECG_OPTIONS,
                    0,
                    "اختر النتيجة المطابقة لتقرير تخطيط القلب.",
                )

        with tests_tab:
            st.caption("نتائج اختبار الجهد والفحوصات القلبية المتخصصة")
            col_a, col_b = st.columns(2, gap="medium")
            with col_a:
                thalach = int(
                    st.number_input(
                        "أقصى معدل لضربات القلب (نبضة/دقيقة)",
                        min_value=40,
                        max_value=230,
                        value=150,
                        step=1,
                        help="أعلى معدل ضربات قلب مسجل أثناء اختبار الجهد.",
                    )
                )
                exang = select_value(
                    "هل ظهرت ذبحة بسبب المجهود؟",
                    YES_NO_OPTIONS,
                    0,
                    "وجود ألم صدري ناتج عن اختبار الجهد.",
                )
                oldpeak = float(
                    st.number_input(
                        "انخفاض مقطع ST الناتج عن المجهود (Oldpeak)",
                        min_value=0.0,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                        format="%.1f",
                        help="قيمة انخفاض ST مقارنة بحالة الراحة.",
                    )
                )
            with col_b:
                slope = select_value(
                    "ميل مقطع ST عند ذروة المجهود",
                    SLOPE_OPTIONS,
                    1,
                    "اتجاه ميل مقطع ST في تخطيط اختبار الجهد.",
                )
                ca = select_value(
                    "عدد الأوعية الرئيسية الملونة بالفلوروسكوبي",
                    CA_OPTIONS,
                    0,
                    "عدد الأوعية الرئيسية الظاهرة في الفحص، من 0 إلى 3.",
                )
                thal = select_value(
                    "نتيجة اختبار الثاليوم",
                    THAL_OPTIONS,
                    3,
                    "اختر النتيجة المسجلة في تقرير اختبار الثاليوم.",
                )

        st.markdown(
            """
            <div class="form-hint">
              راجع القيم قبل التحليل. جودة النتيجة تعتمد على دقة البيانات المدخلة.
            </div>
            """,
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button(
            "تحليل البيانات وإظهار النتيجة",
            type="primary",
            width="stretch",
        )

    values: dict[str, float | int] = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    return submitted, values


def render_empty_result() -> None:
    st.markdown(
        """
        <div class="result-empty">
          <div class="empty-icon" aria-hidden="true">⌁</div>
          <h3>النتيجة ستظهر هنا</h3>
          <p>أكمل الأقسام الثلاثة ثم اضغط زر التحليل.</p>
          <div class="empty-steps">
            <span><b>1</b> أدخل القيم</span>
            <span><b>2</b> حلّل البيانات</span>
            <span><b>3</b> راجع النتيجة</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result: dict[str, Any], values: dict[str, float | int]) -> None:
    probability_percent = result["probability"] * 100
    is_high = result["prediction"] == 1

    tone = "high" if is_high else "low"
    status = "احتمال مرتفع وفق النموذج" if is_high else "احتمال منخفض وفق النموذج"
    icon = "⚠" if is_high else "✓"
    action = (
        "ينصح بعرض النتيجة والقياسات على طبيب مختص لإجراء تقييم سريري كامل."
        if is_high
        else "النتيجة مطمئنة نسبيًا، مع الاستمرار في المتابعة الصحية المعتادة."
    )

    st.markdown(
        f"""
        <div class="result-card result-{tone}">
          <div class="result-topline">
            <span class="result-badge">النتيجة المبدئية</span>
            <span class="result-icon" aria-hidden="true">{icon}</span>
          </div>
          <div class="result-value" dir="ltr">{probability_percent:.1f}%</div>
          <h3>{status}</h3>
          <div class="risk-track" aria-label="الاحتمال التقديري {probability_percent:.1f} بالمئة">
            <span style="width: {probability_percent:.1f}%"></span>
          </div>
          <p class="result-explanation">
            هذه النسبة هي احتمال الفئة الإيجابية كما حسبها النموذج من القيم المدخلة.
          </p>
          <div class="next-action">
            <strong>الخطوة المقترحة</strong>
            <span>{action}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result["outside_range"]:
        fields = "، ".join(result["outside_range"])
        st.warning(
            "بعض القيم خارج النطاق الذي ظهر في بيانات التدريب: "
            f"{fields}. لذلك يجب تفسير النتيجة بحذر أكبر."
        )

    st.markdown(
        """
        <div class="emergency-note">
          <strong>مهم:</strong> إذا كانت الحالة طارئة أو الأعراض مقلقة، لا تنتظر
          نتيجة النظام واطلب رعاية طبية مباشرة.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("عرض ملخص البيانات المدخلة"):
        summary = pd.DataFrame(
            {
                "المؤشر": [FEATURE_LABELS[feature] for feature in EXPECTED_FEATURES],
                "القيمة": [
                    display_value(feature, values[feature])
                    for feature in EXPECTED_FEATURES
                ],
            }
        )
        st.dataframe(summary, hide_index=True, width="stretch")

    st.download_button(
        "تنزيل ملخص النتيجة",
        data=build_report(result, values),
        file_name="BNN_heart_risk_summary.txt",
        mime="text/plain",
        width="stretch",
    )


def render_result_panel(
    result: dict[str, Any] | None, values: dict[str, float | int] | None
) -> None:
    st.markdown(
        """
        <div class="section-title">
          <div class="section-icon">02</div>
          <div>
            <h2>نتيجة التحليل</h2>
            <p>تقدير احتمالي من النموذج، وليس حكمًا أو تشخيصًا طبيًا.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if result is None or values is None:
        render_empty_result()
    else:
        render_result(result, values)


def render_footer() -> None:
    st.markdown(
        """
        <section class="project-facts">
          <div><span>النموذج</span><strong>Logistic Regression</strong></div>
          <div><span>المعالجة</span><strong>Pipeline + StandardScaler</strong></div>
          <div><span>المدخلات</span><strong>13 خاصية سريرية</strong></div>
          <div><span>نطاق الاستخدام</span><strong>تعليمي وبحثي</strong></div>
        </section>
        <footer class="app-footer">
          <strong>تنبيه طبي:</strong>
          هذا النظام نموذج أولي لمشروع تخرج، ولا يشخّص المرض ولا يستبدل الطبيب
          أو الفحوصات الطبية المعتمدة.
        </footer>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def main() -> None:
    apply_styles()
    render_sidebar()
    render_header()

    try:
        model, feature_names = load_model()
    except (
        AttributeError,
        EOFError,
        FileNotFoundError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        st.error("تعذر تحميل نموذج التنبؤ. تأكد من وجود الملف وتوافق إصدار المكتبة.")
        with st.expander("التفاصيل التقنية"):
            st.code(str(exc))
        st.stop()

    st.markdown(
        """
        <div class="privacy-strip">
          <span aria-hidden="true">🔒</span>
          لا نطلب بيانات تعريف شخصية، ولا يحفظ التطبيق سجلًا طبيًا دائمًا.
        </div>
        """,
        unsafe_allow_html=True,
    )

    form_column, result_column = st.columns([1.55, 1], gap="large")

    with form_column:
        submitted, values = render_form()
        if submitted:
            try:
                result = make_prediction(model, feature_names, values)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                st.error("تعذر إتمام التحليل. راجع القيم المدخلة وحاول مرة أخرى.")
                with st.expander("التفاصيل التقنية"):
                    st.code(str(exc))
            else:
                st.session_state["prediction_result"] = result
                st.session_state["prediction_values"] = values

    with result_column:
        render_result_panel(
            st.session_state.get("prediction_result"),
            st.session_state.get("prediction_values"),
        )

    render_footer()


if __name__ == "__main__":
    main()
