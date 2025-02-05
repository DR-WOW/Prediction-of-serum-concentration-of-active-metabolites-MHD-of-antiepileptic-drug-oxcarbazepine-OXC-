import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# 确保 st.set_page_config 在脚本最开始调用
st.set_page_config(layout="wide", page_title="Stacking 模型预测与 SHAP 可视化", page_icon="📊")

# 导入自定义类
from sklearn.base import RegressorMixin, BaseEstimator
from pytorch_tabnet.tab_model import TabNetRegressor

# 定义 TabNetRegressorWrapper 类
class TabNetRegressorWrapper(RegressorMixin, BaseEstimator):
    def __init__(self, **kwargs):
        self.model = TabNetRegressor(**kwargs)
    
    def fit(self, X, y, **kwargs):
        # 将 X 转换为 NumPy 数组
        X = X.values if isinstance(X, pd.DataFrame) else X
        # 将 y 转换为 NumPy 数组并确保它是二维的
        y = y.values if isinstance(y, pd.Series) else y
        y = y.reshape(-1, 1)  # 确保 y 是二维的
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X, **kwargs):
        # 将 X 转换为 NumPy 数组
        X = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X, **kwargs).flatten()  # 将预测结果展平为一维数组

# 加载模型
model_path = "stacking_regressor_model.pkl"
try:
    stacking_regressor = joblib.load(model_path)
    st.success("模型加载成功！")
except Exception as e:
    st.error(f"模型加载失败：{e}")
    raise  # 重新抛出异常以便调试

# 设置页面标题
st.title("📊 Stacking 模型预测与 SHAP 可视化分析")
st.write("""
通过输入特征值进行模型预测，并结合 SHAP 分析结果，了解特征对模型预测的贡献。
""")

# 左侧侧边栏输入区域
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")

# 定义特征输入范围
SEX = st.sidebar.selectbox("性别 Gender(1 = male, 0 = female)", [0, 1])
AGE = st.sidebar.number_input("年龄Age (范围: 0.0-18)", min_value=0.0, max_value=18.0, value=5.0)
WT = st.sidebar.number_input("体重Weight (范围: 0.0-100.0)", min_value=0.0, max_value=100.0, value=25.0)
Single_Dose = st.sidebar.number_input("单次给药剂量/体重Single_Dose/weight (范围: 0.0-60.0)", min_value=0.0, max_value=60.0, value=15.0)
Daily_Dose = st.sidebar.number_input("日总剂量Daily_Dose (范围: 0.0-2400.0)", min_value=0.0, max_value=2400.0, value=450.0)
SCR = st.sidebar.number_input("血清肌酐Serum creatinine (范围: 0.0-150.0)", min_value=0.0, max_value=150.0, value=30.0)
CLCR = st.sidebar.number_input("肌酐清除率Creatinine clearance rate (范围: 0.0-200.0)", min_value=0.0, max_value=200.0, value=90.0)
BUN = st.sidebar.number_input("血尿素氮 (范围: 0.0-50.0)", min_value=0.0, max_value=50.0, value=5.0)
ALT = st.sidebar.number_input("丙氨酸氨基转移酶Alanine aminotransferase (ALT) (范围: 0.0-150.0)", min_value=0.0, max_value=150.0, value=18.0)
AST = st.sidebar.number_input("天冬氨酸氨基转移酶Aspartate transaminase (AST) (范围: 0.0-150.0)", min_value=0.0, max_value=150.0, value=18.0)
CL = st.sidebar.number_input("药物的代谢清除率 Metabolic clearance of drugs (CL)(范围: 0.0-100.0)", min_value=0.0, max_value=100.0, value=3.85)
V = st.sidebar.number_input("药物的表观分布容积(Vd)(范围: 0.0-1000.0)", min_value=0.0, max_value=1000.0, value=10.0)

# 添加预测按钮
predict_button = st.sidebar.button("进行预测")

# 主页面用于结果展示
if predict_button:
    st.header("浓度预测结果(mg/L)")
    try:
        input_array = np.array([SEX, AGE, WT, Single_Dose, Daily_Dose, SCR, CLCR, BUN, ALT, AST, CL, V]).reshape(1, -1)
        prediction = stacking_regressor.predict(input_array)[0]
        st.success(f"预测结果：{prediction:.2f}")
    except Exception as e:
        st.error(f"预测时发生错误：{e}")

# 可视化展示
st.header("SHAP 可视化分析")
st.write("""
以下图表展示了模型的 SHAP 分析结果，包括第一层基学习器、第二层元学习器以及整个 Stacking 模型的特征贡献。
""")

# 第一层基学习器 SHAP 可视化
st.subheader("1. 第一层基学习器")
st.write("基学习器（GBDT、XGBoost、LightGBM、CatBoost、TabNet、LASSO 等6种算法模型）的特征贡献分析。")
first_layer_img = "SHAP Feature Importance of Base Learners in the First Layer of Stacking Model.png"
try:
    img1 = Image.open(first_layer_img)
    st.image(img1, caption="第一层基学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第一层基学习器的 SHAP 图像文件。")

# 第二层元学习器 SHAP 可视化
st.subheader("2. 第二层元学习器")
st.write("元学习器（Linear Regression）的输入特征贡献分析。")
meta_layer_img = "SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor.png"
try:
    img2 = Image.open(meta_layer_img)
    st.image(img2, caption="第二层元学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第二层元学习器的 SHAP 图像文件。")

# 整体 Stacking 模型 SHAP 可视化
st.subheader("3. 整体 Stacking 模型")
st.write("整个 Stacking 模型的特征贡献分析。")
overall_img = "Based on the overall feature contribution analysis of SHAP to the stacking model.png"
try:
    img3 = Image.open(overall_img)
    st.image(img3, caption="整体 Stacking 模型的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到整体 Stacking 模型的 SHAP 图像文件。")

# 页脚
st.markdown("---")
st.header("总结")
st.write("""
通过本页面，您可以：
1. 使用输入特征值进行实时预测。
2. 直观地理解第一层基学习器、第二层元学习器以及整体 Stacking 模型的特征贡献情况。
这些分析有助于深入理解模型的预测逻辑和特征的重要性。
""")
