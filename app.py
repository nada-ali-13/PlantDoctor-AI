"""PlantDoctor AI — Streamlit inference interface. No training is performed here."""
from pathlib import Path
import tempfile

import joblib
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from feature_extractor import extract_features

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

CLASS_AR = {
    "Pepper__bell___Bacterial_spot": ("الفلفل الحلو", "التبقع البكتيري"),
    "Pepper__bell___healthy": ("الفلفل الحلو", "سليم"),
    "Potato___Early_blight": ("البطاطس", "اللفحة المبكرة"),
    "Potato___Late_blight": ("البطاطس", "اللفحة المتأخرة"),
    "Potato___healthy": ("البطاطس", "سليم"),
    "Tomato_Bacterial_spot": ("الطماطم", "التبقع البكتيري"),
    "Tomato_Early_blight": ("الطماطم", "اللفحة المبكرة"),
    "Tomato_Late_blight": ("الطماطم", "اللفحة المتأخرة"),
    "Tomato_Leaf_Mold": ("الطماطم", "عفن الأوراق"),
    "Tomato_Septoria_leaf_spot": ("الطماطم", "تبقع أوراق السبتوريا"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("الطماطم", "العنكبوت الأحمر ذو البقعتين"),
    "Tomato__Target_Spot": ("الطماطم", "التبقع الهدفّي"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("الطماطم", "فيروس تجعد واصفرار أوراق الطماطم"),
    "Tomato__Tomato_mosaic_virus": ("الطماطم", "فيروس موزاييك الطماطم"),
    "Tomato_healthy": ("الطماطم", "سليم"),
}

st.set_page_config(page_title="PlantDoctor AI", page_icon="🌿", layout="centered")
st.markdown("""
<style>
    .block-container {max-width: 850px; padding-top: 2rem;}
    .arabic {direction: rtl; text-align: right; line-height: 1.9;}
    .hero {padding: 1.25rem; border-radius: 18px; background: linear-gradient(120deg,#0e5138,#19794d); color: white; margin-bottom: 1.2rem;}
    .hero h1 {color: white; margin: 0; font-size: 2.1rem;}
    .hero p {color: #e4f5ea; margin: 0.4rem 0 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('''<div class="hero arabic"><h1>🌿 PlantDoctor AI</h1>
<p>مساعد ذكي للتعرف على أمراض أوراق الطماطم والبطاطس والفلفل الحلو</p></div>''', unsafe_allow_html=True)
st.markdown('<div class="arabic">ارفع صورة واضحة لورقة النبات، وسيعرض النموذج أكثر التصنيفات احتمالًا.</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="جاري تحميل النموذج...")
def load_artifacts():
    needed = ["lightgbm_850.joblib", "lgbm_selected_features.npy", "label_encoder.joblib"]
    missing = [f for f in needed if not (ROOT / f).is_file()]
    if missing:
        raise FileNotFoundError("ملفات النموذج غير موجودة: " + ", ".join(missing))
    model = joblib.load(ROOT / needed[0])
    selected = np.load(ROOT / needed[1], allow_pickle=False)
    encoder = joblib.load(ROOT / needed[2])
    if selected.shape != (350,) or len(np.unique(selected)) != 350 or selected.min() < 0 or selected.max() >= 1632:
        raise ValueError("ملف الخصائص المختارة غير متوافق مع النموذج")
    if not np.array_equal(model.classes_, np.arange(len(encoder.classes_))):
        raise ValueError("ترتيب فئات النموذج لا يطابق ملف الترميز")
    return model, selected, encoder

try:
    model, selected, encoder = load_artifacts()
except Exception as exc:
    st.error(f"تعذر تحميل ملفات النموذج: {exc}")
    st.info("فكّي ملفات PlantDoctor_Deployment_Backup.zip وضع ملفات النموذج الثلاثة بجوار app.py، ثم أعد تشغيل التطبيق.")
    st.stop()

uploaded = st.file_uploader("اختار صورة ورقة النبات", type=["jpg", "jpeg", "png", "webp"], help="JPG / JPEG / PNG / WEBP — حتى 10 ميجابايت")

if uploaded is not None:
    raw = uploaded.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        st.error("حجم الصورة أكبر من 10 ميجابايت.")
        st.stop()
    try:
        from io import BytesIO
        with Image.open(BytesIO(raw)) as img:
            img.verify()
        with Image.open(BytesIO(raw)) as img:
            display_image = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("تعذر قراءة الصورة. اختار صورة JPG أو PNG أو WEBP سليمة.")
        st.stop()

    st.image(display_image, caption="الصورة المرفوعة", use_container_width=True)

    if st.button("🔎 تحليل الصورة", type="primary", use_container_width=True):
        # Keep original bytes and extension: cv2.imread matches Kaggle's extractor.
        suffix = Path(uploaded.name).suffix.lower()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(raw)
                temp_path = Path(temp.name)
            features = extract_features(temp_path)
            if features.shape != (1632,) or not np.isfinite(features).all():
                raise ValueError("نتيجة استخراج الخصائص غير صالحة")
            x = np.asarray(features[selected], dtype=np.float32).reshape(1, -1)
            probs = np.asarray(model.predict_proba(x)[0], dtype=float)
            order = np.argsort(probs)[::-1]
            class_ids = np.asarray(model.classes_, dtype=int)
            winner = int(class_ids[order[0]])
            raw_label = str(encoder.inverse_transform([winner])[0])
            plant, condition = CLASS_AR.get(raw_label, ("نبات غير محدد", raw_label))

            st.success("اكتمل تحليل الصورة")
            col1, col2 = st.columns(2)
            col1.metric("🌱 نوع النبات المتوقع", plant)
            col2.metric("🩺 الحالة المتوقعة", condition)
            st.metric("أعلى احتمال من النموذج", f"{probs[order[0]] * 100:.1f}%")
            st.caption("هذه احتمالات مقدّرة من النموذج وليست ضمانًا لصحة التشخيص، ولم تُختبر معايرتها إحصائيًا.")
            st.markdown("#### أبرز ثلاثة احتمالات")
            for rank in order[:3]:
                label = str(encoder.inverse_transform([int(class_ids[rank])])[0])
                p_name, disease = CLASS_AR.get(label, ("", label))
                st.write(f"**{p_name} — {disease}**: {probs[rank] * 100:.1f}%")
                st.progress(float(np.clip(probs[rank], 0.0, 1.0)))

            if probs[order[0]] < 0.65:
                st.warning("النتيجة غير حاسمة وفق مؤشر احتمالي أولي. حاول استخدام صورة أوضح، ولا تعتمد على هذا التوقع وحده.")
            st.info(" ")
        except Exception as exc:
            st.error(f"حدث خطأ أثناء تحليل الصورة: {exc}")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

st.divider()
st.caption("PlantDoctor AI | Traditional ML: HSV + RGB + LBP + HOG → LightGBM | Independent PlantVillage test accuracy: 95.18%")
st.caption("نتائج الاختبار تخص مجموعة PlantVillage؛ لا تمثل ضمانًا للدقة على صور الحقول أو الهواتف.")
