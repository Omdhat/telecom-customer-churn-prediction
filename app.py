"""Streamlit dashboard for the Telecom Customer Churn Prediction project."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))          
from features import INPUT_COLUMNS  # noqa: E402

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = BASE_DIR / "churn_data.csv"

st.set_page_config(page_title="Telecom Customer Churn Prediction", page_icon="📉", layout="wide")
st.markdown("""
<style>
.result-card {padding: 1.2rem 1.5rem; border-radius: 12px; font-size: 1.4rem; font-weight: 600;}
.churn {background: #fdecea; color: #b42318; border: 1px solid #f5c2c0;}
.stay {background: #e8f6ee; color: #1a7f45; border: 1px solid #b7e2c8;}
</style>
""", unsafe_allow_html=True)

FIELDS = [
    ("Account length (days)", "accountlength", "int"),
    ("International plan", "internationalplan", "yesno"),
    ("Voicemail plan", "voicemailplan", "yesno"),
    ("Voicemail messages", "numbervmailmessages", "int"),
    ("Day minutes", "totaldayminutes", "float"),
    ("Day calls", "totaldaycalls", "int"),
    ("Evening minutes", "totaleveminutes", "float"),
    ("Evening calls", "totalevecalls", "int"),
    ("Night minutes", "totalnightminutes", "float"),
    ("Night calls", "totalnightcalls", "int"),
    ("International minutes", "totalintlminutes", "float"),
    ("International calls", "totalintlcalls", "int"),
    ("Customer service calls", "numbercustomerservicecalls", "int"),
]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_customers():
    df = pd.read_csv(DATA_PATH)
    df.columns = ["CustomerID" if c.strip().lower() == "customerid" else c.strip().lower()
                  for c in df.columns]
    return df.set_index("CustomerID") if "CustomerID" in df.columns else None


@st.cache_data
def load_test_ids():
    path = OUTPUT_DIR / "test_customer_ids.csv"
    return set(pd.read_csv(path)["CustomerID"]) if path.exists() else None


def is_yes(value):
    return str(value).strip().lower() in ("true", "yes", "1")


def yes_no(value):
    return "Yes" if is_yes(value) else "No"


def predict_one(bundle, features):
    probability = float(bundle["pipeline"].predict_proba(features)[0, 1])
    return probability >= bundle["threshold"], probability


def result_card(caption, text, is_churn):
    st.caption(caption)
    style = "churn" if is_churn else "stay"
    st.markdown(f'<div class="result-card {style}">{text}</div>', unsafe_allow_html=True)


def show_probabilities(bundle, probability):
    left, right = st.columns([2, 1])
    left.caption("Predicted churn probability")
    left.progress(min(max(probability, 0.0), 1.0))
    right.metric("Predicted Churn Probability", f"{probability:.2%}")
    right.metric("No-Churn Probability", f"{1 - probability:.2%}")
    st.caption(f"Model: {bundle['model_name']}. 'Likely to Churn' means the churn probability is at or "
               f"above the decision threshold ({bundle['threshold']:.2f}).")


def show_images(*filenames):
    for col, name in zip(st.columns(len(filenames)), filenames):
        with col:
            if (OUTPUT_DIR / name).exists():
                st.image(str(OUTPUT_DIR / name))
            else:
                st.warning(f"{name} not found. Run eda.py and model.py first.")

st.sidebar.title("📉 Churn Predictor")
page = st.sidebar.radio("Navigation", ["🏠 Churn Prediction", "📊 EDA & Visualizations",
                                       "🤖 Model Comparison"])

if page == "🏠 Churn Prediction":
    st.title("🏠 Customer Churn Prediction")

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python model.py` first.")
        st.stop()
    bundle = load_model()

    mode = st.radio("Prediction Mode", ["New Customer", "Existing Customer"], horizontal=True)
    st.divider()

    if mode == "New Customer":
        st.subheader("Account & Plans")
        c1, c2, c3, c4 = st.columns(4)
        account_length = c1.slider("Account length (days)", min_value=1, max_value=365, value=90)
        intl_plan = c2.segmented_control("International plan", ["No", "Yes"])
        vm_plan = c3.segmented_control("Voicemail plan", ["No", "Yes"])
        vm_messages = c4.slider("Voicemail messages", min_value=0, max_value=60, value=0, disabled=(vm_plan == "No"))
        if vm_plan == "No":
            vm_messages = 0

        st.subheader("Call Usage")
        d1, d2, d3, d4 = st.columns(4)
        day_min = d1.number_input("Day minutes", 0.0, 400.0, 180.0, step=1.0)
        day_calls = d1.slider("Day calls", min_value=0, max_value=200, value=100)
        eve_min = d2.number_input("Evening minutes", 0.0, 400.0, 200.0, step=1.0)
        eve_calls = d2.slider("Evening calls", min_value=0, max_value=200, value=100)
        night_min = d3.number_input("Night minutes", 0.0, 400.0, 200.0, step=1.0)
        night_calls = d3.slider("Night calls", min_value=0, max_value=200, value=100)
        intl_min = d4.number_input("International minutes", 0.0, 25.0, 10.0, step=0.1)
        intl_calls = d4.slider("International calls", min_value=0, max_value=25, value=4)

        st.subheader("Customer Service")
        service_calls = st.slider("Customer service calls", 0, 10, 1)

        if st.button("Predict Churn", type="primary"):
            row = pd.DataFrame([{
                "accountlength": account_length,
                "internationalplan": intl_plan == "Yes",
                "voicemailplan": vm_plan == "Yes",
                "numbervmailmessages": vm_messages,
                "totaldayminutes": day_min, "totaldaycalls": day_calls,
                "totaleveminutes": eve_min, "totalevecalls": eve_calls,
                "totalnightminutes": night_min, "totalnightcalls": night_calls,
                "totalintlminutes": intl_min, "totalintlcalls": intl_calls,
                "numbercustomerservicecalls": service_calls,
            }])[INPUT_COLUMNS]
            will_churn, probability = predict_one(bundle, row)

            st.divider()
            result_card("Model prediction",
                        "⚠️ Likely to Churn" if will_churn else "✅ Unlikely to Churn", will_churn)
            show_probabilities(bundle, probability)

    else:
        customers = load_customers()
        if customers is None:
            st.error("churn_data.csv has no CustomerID column yet. Run `python model.py` once to create it.")
            st.stop()
        test_ids = load_test_ids()

        left, right = st.columns(2)
        only_test = left.checkbox("Show only test-set customers", disabled=(test_ids is None),
                                  help="Test-set customers were never seen by the model during training.")
        if test_ids is None:
            left.warning("outputs/test_customer_ids.csv not found. Run `python model.py`.")
        search = right.text_input("Search Customer ID (optional)", placeholder="e.g. CUST0042")

        id_list = list(customers.index)
        if only_test:
            id_list = [i for i in id_list if i in test_ids]
        if search:
            id_list = [i for i in id_list if search.strip().upper() in i]
        if not id_list:
            st.warning("No customers match your search.")
            st.stop()

        customer_id = st.selectbox(f"Select Customer ({len(id_list)} available - you can also type in the box)",
                                   id_list)
        customer = customers.loc[customer_id]
        in_test_set = test_ids is not None and customer_id in test_ids
        split_label = "Test Set Customer" if in_test_set else "Training Set Customer"

        st.subheader("Customer Information")
        st.caption("Loaded from the dataset. Read-only, so the prediction always uses the real values.")
        boxes = st.columns(4)
        for i, (label, column, kind) in enumerate(FIELDS):
            box, value, key = boxes[i % 4], customer[column], f"{customer_id}_{column}"
            if kind == "yesno":
                box.selectbox(label, ["No", "Yes"], index=int(is_yes(value)), disabled=True, key=key)
            elif kind == "int":
                box.number_input(label, value=int(value), disabled=True, key=key)
            else:
                box.number_input(label, value=float(value), format="%.1f", disabled=True, key=key)

        if st.button("🔍 Analyze Customer", type="primary"):
            will_churn, probability = predict_one(bundle, customers.loc[[customer_id], INPUT_COLUMNS])
            actual_churned = is_yes(customer["churn"])

            st.divider()
            with st.container(border=True):                
                st.markdown(f"### Customer {customer_id}")
                st.caption(f"{'🧪' if in_test_set else '📚'} {split_label}")
                summary = [
                    ("Account Length", f"{int(customer['accountlength'])} days"),
                    ("International Plan", yes_no(customer["internationalplan"])),
                    ("Voicemail Plan", yes_no(customer["voicemailplan"])),
                    ("Customer Service Calls", int(customer["numbercustomerservicecalls"])),
                    ("Daily Minutes", f"{customer['totaldayminutes']:.1f}"),
                    ("Evening Minutes", f"{customer['totaleveminutes']:.1f}"),
                    ("Night Minutes", f"{customer['totalnightminutes']:.1f}"),
                    ("International Minutes", f"{customer['totalintlminutes']:.1f}"),
                ]
                metric_boxes = st.columns(4)
                for i, (label, value) in enumerate(summary):
                    metric_boxes[i % 4].metric(label, value)

            st.subheader("Actual vs Predicted")
            a, b = st.columns(2)
            with a:
                result_card("Actual status in dataset",
                            "Churned" if actual_churned else "Did Not Churn", actual_churned)
            with b:
                result_card("Model prediction",
                            "Likely to Churn" if will_churn else "Unlikely to Churn", will_churn)
            st.write("")
            show_probabilities(bundle, probability)

            if in_test_set:
                if will_churn == actual_churned:
                    st.success("✅ Correct prediction on an unseen test-set customer.")
                else:
                    st.error("❌ Incorrect prediction on an unseen test-set customer.")
            else:
                st.info("This customer was in the training data, so the model has already learned from "
                        "this row. The result is NOT evidence of generalization. Tick 'Show only "
                        "test-set customers' to demonstrate real performance.")


elif page == "📊 EDA & Visualizations":
    st.title("📊 EDA & Visualizations")

    st.header("Exploratory Data Analysis")
    show_images("churn_distribution.png", "churn_vs_service_calls.png")
    show_images("churn_vs_international_plan.png", "churn_vs_voicemail_plan.png")
    show_images("usage_vs_churn.png")
    show_images("correlation_heatmap.png")

    st.header("Model Evaluation (untouched test set)")
    show_images("confusion_matrix.png", "roc_curve.png")

    st.header("Threshold Tuning (validation predictions)")
    show_images("threshold_tuning.png")

else:
    st.title("🤖 Model Comparison")

    if not (OUTPUT_DIR / "model_comparison.csv").exists():
        st.error("outputs/model_comparison.csv not found. Run `python model.py` first.")
        st.stop()

    comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    info = json.loads((OUTPUT_DIR / "test_metrics.json").read_text())
    params = json.loads((OUTPUT_DIR / "best_params.json").read_text())
    best = info["best_model"]

    st.success(f"Model selected by cross-validation: **{best}** (decision threshold {info['threshold']:.2f})")

    st.subheader("Final result on the untouched test set")
    metric_boxes = st.columns(5)
    for box, metric in zip(metric_boxes, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        box.metric(metric, f"{info[metric]:.3f}")
    st.caption(f"Test set: {info['test_rows']} customers, used once for the selected model only.")

    st.subheader("Cross-validation comparison (training data only)")
    table = comparison.copy()
    table.insert(0, "Selected", ["✅" if m == best else "" for m in table["Model"]])
    st.dataframe(table.round(4), hide_index=True)
    st.caption("CV F1 = mean 5-fold F1 at threshold 0.5 during tuning. The other columns use 5-fold "
               "out-of-fold predictions at each model's tuned threshold. No test data was used.")

    for box, metric in zip(st.columns(3), ["Accuracy", "F1", "ROC-AUC"]):
        with box:
            st.markdown(f"**{metric}**")
            st.bar_chart(comparison.set_index("Model")[metric])

    if (OUTPUT_DIR / "feature_engineering_comparison.csv").exists():
        st.subheader("Effect of feature engineering (5-fold CV F1, default settings)")
        st.dataframe(pd.read_csv(OUTPUT_DIR / "feature_engineering_comparison.csv"), hide_index=True)

    st.subheader("Best hyperparameters")
    for name in comparison["Model"]:
        with st.expander(f"{name} (selected)" if name == best else name):
            st.json(params[name])
