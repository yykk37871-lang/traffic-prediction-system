import pandas as pd

# ==========================================
# 1. 【核心补丁】解决库兼容性问题
# ==========================================
try:
    import pandas.core.strings as pd_strings
    if not hasattr(pd_strings, 'StringMethods'):
        from pandas.core.strings.accessor import StringMethods
        pd_strings.StringMethods = StringMethods
except:
    pass

import streamlit as st
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
import joblib
import os

# ==========================================
# 2. 全局配置与【云端字体修复】
# ==========================================
st.set_page_config(page_title="智慧交通拥堵预测系统", layout="wide")

# 修复负号方格与中文字体（优先匹配 Linux 云端字体）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

FEATURE_COLS = ['hour', 'day_of_week', 'is_holiday', 'temp', 'weather_enc', 'lag_1', 'lag_2', 'rolling_mean_3h']
CLASS_NAMES = ['重度拥堵', '高拥堵', '正常路况', '低拥堵']
CLASS_COLORS = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71']
WEATHER_MAP = {"晴朗": 1, "多云": 0, "有雨": 6, "小雨": 2, "有雾": 5, "雷暴": 10, "下雪": 8}

@st.cache_resource
def load_resources():
    return joblib.load('traffic_model.pkl'), joblib.load('background_data.pkl')

model, background = load_resources()

# --- 流量联动仿真函数 ---
def simulate_traffic_volume(hour, day, holiday, temp, weather_name):
    peak_morning = 4200 * np.exp(-((hour - 8)**2) / 3.5)
    peak_evening = 4800 * np.exp(-((hour - 17)**2) / 5)
    base_flow = 1500 + peak_morning + peak_evening
    weekday_multiplier = 0.75 if day >= 5 else 1.0
    holiday_multiplier = 0.6 if holiday == 1 else 1.0
    weather_impact_map = {"晴朗": 1.0, "多云": 1.0, "有雨": 0.85, "小雨": 0.95, "有雾": 0.8, "雷暴": 0.6, "下雪": 0.5}
    weather_multiplier = weather_impact_map.get(weather_name, 1.0)
    temp_multiplier = 0.85 if temp < 265 else 1.0
    return int(base_flow * weekday_multiplier * holiday_multiplier * weather_multiplier * temp_multiplier)

# ==========================================
# 3. 侧边栏交互
# ==========================================
st.sidebar.header("控制中心")
input_hour = st.sidebar.slider("时间点 (Hour)", 0, 23, 8)
input_day = st.sidebar.selectbox("星期 (Day)", options=[0,1,2,3,4,5,6], format_func=lambda x: ['周一','周二','周三','周四','周五','周六','周日'][x])
input_holiday = st.sidebar.radio("是否节假日 (Holiday)", options=[0, 1], format_func=lambda x: "是" if x==1 else "否")
input_temp = st.sidebar.slider("实时气温 (Temp K)", 240, 320, 290)
weather_text = st.sidebar.selectbox("天气状况 (Weather)", options=list(WEATHER_MAP.keys()))

auto_v = simulate_traffic_volume(input_hour, input_day, input_holiday, input_temp, weather_text)
input_lag1, input_lag2 = int(auto_v * 0.96), int(auto_v * 0.92)
input_rolling = (auto_v + input_lag1 + input_lag2) / 3

# ==========================================
# 4. 核心计算
# ==========================================
input_data = pd.DataFrame([[input_hour, input_day, input_holiday, input_temp, WEATHER_MAP[weather_text], input_lag1, input_lag2, input_rolling]], columns=FEATURE_COLS)

# 预测概率
pred_probs = model.predict(input_data)
pred_class = np.argmax(pred_probs[0])

# ==========================================
# 5. 前端渲染
# ==========================================
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🚦 智慧交通拥堵预测系统</h1>", unsafe_allow_html=True)
st.divider()

col_left, col_right = st.columns([1, 2])

# --- 左侧：核心指标 ---
with col_left:
    st.subheader("预测报告")
    st.markdown(f"""<div style='padding:20px; border-radius:10px; background-color:{CLASS_COLORS[pred_class]}; color:white; text-align:center;'>
        <h1 style='margin:0; font-size:40px;'>{CLASS_NAMES[pred_class]}</h1>
        <p style='margin:0;'>判定置信度: {np.max(pred_probs)*100:.2f}%</p></div>""", unsafe_allow_html=True)
    st.write("---")
    st.write("**参数联动看板：**")
    st.metric("预估实时流量", f"{auto_v} 辆/h")
    st.metric("气温状态", f"{input_temp} K")
    st.metric("天气负荷", weather_text)
    st.caption("注：归因分析基于 SHAP 归因理论实时计算生成。")

# --- 右侧：智能诊断报告 ---
with col_right:
    # --- 【核心算法修复：概率空间归因】 ---
    # 我们让 SHAP 解释 predict 函数的输出（概率），而非原始的 logit
    explainer = shap.Explainer(lambda x: model.predict(x), background)
    shap_values = explainer(input_data)
    
    # 动态切片：只提取当前预测出来的那个类别的贡献值
    vals = shap_values.values[0, :, pred_class]
    base = shap_values.base_values[0, pred_class]
    
    cn_names = ['时间', '星期', '节假日', '气温', '天气', '联动流量(Lag1)', '联动流量(Lag2)', '联动均值']
    
    st.write("**诊断结论报告：**")
    top_indices = np.argsort(np.abs(vals))[::-1][:3]
    report_items = []
    for i in top_indices:
        report_items.append(f"**{cn_names[i]}**：贡献 {vals[i]:+.2f}，表现为{'【风险推高】↑' if vals[i]>0 else '【压力缓解】↓'}")
    
    st.info(f"""**诊断摘要：** 当前判定为 **{CLASS_NAMES[pred_class]}**。其核心逻辑在于：
1. {report_items[0]}  
2. {report_items[1]}  
3. {report_items[2]}""")

    st.write("**因果归因推导路径：**")
    # 构造 Explanation 对象
    exp_obj = shap.Explanation(values=vals, base_values=base, data=input_data.iloc[0], feature_names=cn_names)
    
    # 缩小高度以适配页面，增加 DPI 保证清晰度
    fig, ax = plt.subplots(figsize=(10, 4.5)) 
    shap.plots.waterfall(exp_obj, show=False)
    plt.title(f"预测判定逻辑链条拆解", fontsize=12, pad=10)
    st.pyplot(fig)
