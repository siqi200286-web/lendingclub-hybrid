import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix



 1. Streamlit 기본 설정

st.set_page_config(
    page_title="LendingClub Hybrid Model",
    layout="wide"
)

st.title(" Lending Club Hybrid Model (Logistic Regression + Gradient Boosting)")
st.write("Lending Club 데이터를 활용하여 혼합 모델(Hybrid Model)을 구축하고, 학습 결과와 예측 데모를 제공합니다.")



2. 데이터 불러오기

DATA_PATH = "LC_clean (1).csv"   # CSV 파일 이름

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # 날짜 → 년/월로 변환
    df["Application Date"] = pd.to_datetime(df["Application Date"])
    df["AppYear"] = df["Application Date"].dt.year
    df["AppMonth"] = df["Application Date"].dt.month

    # 원본 날짜 컬럼 삭제
    df = df.drop(columns=["Application Date"])

    return df

df = load_data(DATA_PATH)

st.subheader(" 데이터 미리보기")
st.write(df.head())
st.write(f"총 데이터 수: {df.shape[0]}개, 특징 수: {df.shape[1] - 1}개 (타깃 변수 제외)")



 3. 특징 정의

target_col = "Approved"
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

st.write(" **수치형 변수:**", numeric_features)
st.write(" **범주형 변수:**", categorical_features)



 4. 전처리 및 모델 정의

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 기본 모델 1: Logistic Regression
log_reg = LogisticRegression(max_iter=1000)

# 기본 모델 2: Gradient Boosting
gb_clf = GradientBoostingClassifier(random_state=42)

# Hybrid Model: Soft Voting
hybrid_clf = VotingClassifier(
    estimators=[
        ("log_reg", log_reg),
        ("gb", gb_clf),
    ],
    voting="soft"
)

# 사이드바 설정
st.sidebar.header(" 모델 설정")
model_choice = st.sidebar.selectbox(
    "모델 선택:",
    ("Logistic Regression", "Gradient Boosting", "Hybrid (Logistic + GBM)")
)

if model_choice == "Logistic Regression":
    chosen_model = log_reg
elif model_choice == "Gradient Boosting":
    chosen_model = gb_clf
else:
    chosen_model = hybrid_clf

test_size = st.sidebar.slider("테스트 데이터 비율 (test_size)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=42)



# 5. 모델 학습 및 성능 출력

if st.button(" 모델 학습 시작"):
    with st.spinner("모델 학습 중입니다..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", chosen_model)
        ])

        clf.fit(X_train, y_train)

        # 예측
        y_pred = clf.predict(X_test)

        if hasattr(clf.named_steps["model"], "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
        else:
            y_proba = None
            roc = None

        acc = accuracy_score(y_test, y_pred)

        st.subheader(" 모델 성능 결과")
        st.write(f"**선택된 모델:** {model_choice}")
        st.write(f"**Accuracy:** {acc:.4f}")
        if roc is not None:
            st.write(f"**ROC-AUC:** {roc:.4f}")

        st.write("**Classification Report**")
        st.text(classification_report(y_test, y_pred, digits=4))

        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        st.write(pd.DataFrame(
            cm,
            index=["True 0", "True 1"],
            columns=["Pred 0", "Pred 1"]
        ))

        st.success("모델 학습 완료 ")

        # 예측에 사용하기 위해 저장
        st.session_state["trained_clf"] = clf
        st.session_state["numeric_features"] = numeric_features
        st.session_state["categorical_features"] = categorical_features



# 6. 단일 레코드 예측 Demo

st.subheader(" 단일 데이터 예측 데모")

if "trained_clf" not in st.session_state:
    st.info("먼저 위에서 모델 학습을 진행하세요.")
else:
    clf = st.session_state["trained_clf"]
    numeric_features = st.session_state["numeric_features"]
    categorical_features = st.session_state["categorical_features"]

    st.write("아래 항목을 입력하면 승인 여부를 예측합니다:")

    input_data = {}

    cols = st.columns(2)

    # 수치형 입력
    with cols[0]:
        for col in numeric_features:
            default_val = float(df[col].median())
            input_data[col] = st.number_input(
                f"{col}",
                value=default_val
            )

    # 범주형 입력
    with cols[1]:
        for col in categorical_features:
            options = df[col].astype(str).unique().tolist()
            most_common = df[col].value_counts().idxmax()
            input_data[col] = st.selectbox(
                f"{col}",
                options=options,
                index=options.index(str(most_common))
            )

    if st.button(" 승인 여부 예측"):
        input_df = pd.DataFrame([input_data])
        proba = clf.predict_proba(input_df)[0, 1]
        pred = clf.predict(input_df)[0]

        st.write("**예측 결과:**")
        st.write(f"승인 확률 (P=1) = **{proba:.4f}**")
        if pred == 1:
            st.success("예측 결과: 승인될 가능성이 높습니다 ")
        else:
            st.warning("예측 결과: 승인되지 않을 가능성이 높습니다 ")
