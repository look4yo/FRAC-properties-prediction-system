import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import dill

# 显示中文和负号
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 加载模型
MQ_voting_model = joblib.load('MQ_voting_model.joblib')
MS_voting_model = joblib.load('MS_voting_model.joblib')
FV_voting_model = joblib.load('FV_voting_model.joblib')

# ✅ 使用 dill 加载 explainer
with open('MQ_explainer_file.dill', 'rb') as f:
    MQ_explainer_model = dill.load(f)

with open('MS_explainer_file.dill', 'rb') as f:
    MS_explainer_model = dill.load(f)

with open('FV_explainer_file.dill', 'rb') as f:
    FV_explainer_model = dill.load(f)

scaler = joblib.load('scaler.joblib')
# 特征列表（与训练时一致）
features = ['Pe', 'Du', 'SP', 'AC', 'AV', 'VMA', 'VFA', 'Ag2.36', 'Ag4.75', 'Ag9.5',
            'FT', 'FC', 'FL', 'TS', 'MT']

# 设置页面
st.set_page_config(page_title="Multi-Target Prediction System", layout="wide")
st.markdown(
    "<div style='background-color:#007BFF;padding:1px;border-radius:5px;text-align:center;'>"
    "<h3 style='color:white;margin:0;font-size:30px;'>Marshall Quotient(MQ), Stability(MS) & Flow Value(FV) Prediction System</h3>"
    "</div>",
    unsafe_allow_html=True
)
# 1、特征输入
st.markdown(
    "<div style='background-color:orange;padding:1px 10px;border-radius:5px;display:inline-block;margin-top:20px;margin-bottom:20px;'>"
    "<h2 style='color:white;margin:0;font-size:30px;text-align:left;'>Input parameters</h2>"
    "</div>",
    unsafe_allow_html=True
)
# 定义纤维类型映射
fiber_type_mapping = {
    1: "Glass fiber",
    2: "Mineral fiber",
    3: "Plastic fiber",
    4: "Steel fiber",
    5: "Carbon fiber",
    6: "Bio-fiber"
}

st.html("""
<style>
    [data-testid="stNumberInput"] input {
        font-size: 30px !important;
    }
</style>
""")


def styled_number_input(label, min_value, max_value, value, step, label_fontsize="20px", key=None):
    # 使用 st.markdown 显示带样式的 label
    st.markdown(f"<span style='font-size: {label_fontsize}; font-weight: normal;'>{label}</span>",
                unsafe_allow_html=True)

    # 使用 st.number_input，并隐藏原始 label
    return st.number_input(
        " ",  # 空字符串会导致 Streamlit 报错，用空格替代
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        key=key,
        label_visibility="collapsed"
    )


st.markdown("""
<style>
    /* 设置 st.selectbox 下拉框中选中文字和选项字体大小 */
    .stSelectbox div[data-baseweb="select"] > div {
        font-size: 22px;
    }
</style>
""", unsafe_allow_html=True)


def styled_selectbox(label, options, index=0, format_func=None, key=None, label_fontsize="24px", **kwargs):
    # 使用 markdown 显示自定义 label
    st.markdown(f"<span style='font-size: {label_fontsize}; font-weight: normal;'>{label}</span>",
                unsafe_allow_html=True)

    # 使用 st.selectbox，并隐藏原生 label
    return st.selectbox(
        " ",  # 用空字符串替代 label
        options=options,
        index=index,
        format_func=format_func,
        key=key,
        label_visibility="collapsed",
        **kwargs
    )


# 创建三列布局
col1, col2, col3 = st.columns(3)

with col1:
    AC = styled_number_input("Asphalt content (AC) [wei.%]", 2.0, 15.0, 5.5, 0.1,
                             label_fontsize="24px", key="ac")
    Pe = styled_number_input("Penetration (Pe) [0.1 mm]", 30.0, 110.0, 68.0, 0.1,
                             label_fontsize="24px", key="pe")
    Du = styled_number_input("Ductility (Du) [cm]", 20.0, 200.0, 150.0, 1.0,
                             label_fontsize="24px", key="du")
    SP = styled_number_input("Softening point (SP) [℃]", 40.0, 60.0, 49.0, 0.1,
                             label_fontsize="24px", key="sp")
    Ag2_36 = styled_number_input("Aggregate passing 2.36mm (Ag2.36) [%]", 0.0, 100.0, 47.0, 0.01,
                                 label_fontsize="24px", key="ag236")

with col2:
    Ag4_75 = styled_number_input("Aggregate passing 4.75mm (Ag4.75) [%]", 0.0, 100.0, 60.0, 0.01,
                                 label_fontsize="24px", key="ag475")
    Ag9_5 = styled_number_input("Aggregate passing 9.5mm (Ag9.5) [%]", 0.0, 100.0, 80.0, 0.01,
                                label_fontsize="24px", key="ag95")
    AV = styled_number_input("Air voids (AV) [%]", 0.0, 25.0, 2.8, 0.01,
                             label_fontsize="24px", key="av")
    VMA = styled_number_input("Voids in mineral aggregate (VMA) [%]", 5.0, 70.0, 14.63, 0.01,
                              label_fontsize="24px", key="vma")
    VFA = styled_number_input("Voids filled with asphalt (VFA) [%]", 10.0, 100.0, 80.0, 0.01,
                              label_fontsize="24px", key="vfa")

with col3:
    FT = styled_selectbox(
        "Fiber type (FT)",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: fiber_type_mapping[x],
        index=0,
        key="ft_select",
        label_fontsize="24px"
    )
    FC = styled_number_input("Fiber content (FC) [wei.%]", 0.0, 7.0, 0.6, 0.01,
                             label_fontsize="24px", key="fc")
    FL = styled_number_input("Fiber length (FL) [mm]", 0.0, 130.0, 12.0, 0.01,
                             label_fontsize="24px", key="fl")
    TS = styled_number_input("Tensile strength (TS) [MPa]", 0.0, 10000.0, 3000.0, 0.1,
                             label_fontsize="24px", key="ts")
    MT = styled_number_input("Melting temperature (MT) [℃]", 50.0, 5000.0, 550.0, 1.0,
                             label_fontsize="24px", key="mt")

# 构建输入数据
input_data = [[Pe, Du, SP, AC, AV, VMA, VFA, Ag2_36, Ag4_75, Ag9_5,
               FT, FC, FL, TS, MT]]

st.markdown("""
<style>
    div.stButton > button {
        padding: 10px 20px !important; /* 可选：调整按钮内边距 */
        height: 50px !important;     /* 可选：调整按钮高度 */
        background-color: #E0F7FF !important; /* 背景色改为淡蓝色 */
        font-weight: bold !important; /* 字体加粗 */
    }
</style>
""", unsafe_allow_html=True)
if st.button("Predict three objectives", use_container_width=True):
    # 创建状态提示框
    status_placeholder = st.empty()
    status_placeholder.info("Calculating...")

    # 标准化输入
    scaled_input = scaler.transform(input_data)

    # 分别预测三个目标
    mq_prediction = MQ_voting_model.predict(scaled_input)[0]
    ms_prediction = MS_voting_model.predict(scaled_input)[0]
    fv_prediction = FV_voting_model.predict(scaled_input)[0]

    # 存储结果
    st.session_state['mq_prediction'] = mq_prediction
    st.session_state['ms_prediction'] = ms_prediction
    st.session_state['fv_prediction'] = fv_prediction
    st.session_state['input_data'] = input_data

# 2、显示结果
st.markdown(
    "<div style='background-color:orange;padding:1px 10px;border-radius:5px;display:inline-block;margin-top:20px;margin-bottom:20px;'>"
    "<h2 style='color:white;margin:0;font-size:30px;text-align:left;'>Prediction results</h2>"
    "</div>",
    unsafe_allow_html=True
)
if 'mq_prediction' in st.session_state:
    col_mq, col_ms, col_fv = st.columns(3)

    with col_mq:
        st.markdown("<p style='font-size:25px; font-weight:bold;'>Marshall quotient (MQ)</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;text-align:center;'>"
            f"<span style='font-size:25px; font-weight:bold;'>{st.session_state['mq_prediction']:.2f} kN/mm</span></div>",
            unsafe_allow_html=True
        )

    with col_ms:
        st.markdown("<p style='font-size:25px; font-weight:bold;'>Marshall stability (MS)</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;text-align:center;'>"
            f"<span style='font-size:25px; font-weight:bold;'>{st.session_state['ms_prediction']:.2f} kN</span></div>",
            unsafe_allow_html=True
        )

    with col_fv:
        st.markdown("<p style='font-size:25px; font-weight:bold;'>Marshall flow value (FV)</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;text-align:center;'>"
            f"<span style='font-size:25px; font-weight:bold;'>{st.session_state['fv_prediction']:.2f} mm</span></div>",
            unsafe_allow_html=True
        )

    # 特征对应的单位（映射字典）
    feature_units = {
        'Pe': '0.1 mm',
        'Du': 'cm',
        'SP': '℃',
        'AC': 'wei.%',
        'AV': '%',
        'VMA': '%',
        'VFA': '%',
        'Ag2.36': '%',
        'Ag4.75': '%',
        'Ag9.5': '%',
        'FT': '',
        'FC': 'wei.%',
        'FL': 'mm',
        'TS': 'MPa',
        'MT': '℃'
    }

    # 显示输入参数汇总
    st.subheader("Input parameters summary")
    # 构造 DataFrame
    input_df = pd.DataFrame(st.session_state['input_data'], columns=features)
    # 添加单位到列名（仅显示时）
    input_df.rename(columns={f: f"{f} [{feature_units[f]}]" for f in input_df.columns}, inplace=True)
    # 显示表格（加大字体）
    st.dataframe(
        input_df.style.format("{:.2f}").set_properties(**{'font-size': '25px'}),
        use_container_width=True
    )

# 3、SHAP分析
st.markdown(
    "<div style='background-color:orange;padding:1px 5px;border-radius:5px;display:inline-block;margin-top:20px;margin-bottom:20px;'>"
    "<h2 style='color:white;margin:0;font-size:30px;text-align:left;'>SHAP explanation</h2>"
    "</div>",
    unsafe_allow_html=True
)

if 'mq_prediction' in st.session_state:
    # 获取原始输入数据（不经过标准化）
    original_input = np.array(st.session_state['input_data'])

    # 创建三列展示每个模型的 SHAP 图
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### SHAP force plot - MQ")
        mq_shap_values = MQ_explainer_model.shap_values(original_input)[0]
        plt.rcParams.update({
            'font.size': 30,
            'xtick.labelsize': 30,
        })

        # 使用 SHAP 内部创建的 figure，不手动传入 fig 或 ax
        force_fig = shap.force_plot(
            MQ_explainer_model.expected_value[0],
            mq_shap_values,
            feature_names=features,
            matplotlib=True,
            show=False
        )

        # 在 Streamlit 中显示 figure，并在显示后清除
        st.pyplot(force_fig, clear_figure=True)

    with col2:
        st.write("### SHAP force plot - MS")
        ms_shap_values = MS_explainer_model.shap_values(original_input)[0]

        force_fig = shap.force_plot(
            MS_explainer_model.expected_value[0],
            ms_shap_values,
            feature_names=features,
            matplotlib=True,
            show=False
        )

        st.pyplot(force_fig, clear_figure=True)

    with col3:
        st.write("### SHAP force plot - FV")
        fv_shap_values = FV_explainer_model.shap_values(original_input)[0]

        force_fig = shap.force_plot(
            FV_explainer_model.expected_value[0],
            fv_shap_values,
            feature_names=features,
            matplotlib=True,
            show=False
        )

        st.pyplot(force_fig, clear_figure=True)

    # 添加瀑布图（修改后，不再使用 ax 参数）
    col4, col5, col6 = st.columns(3)

    with col4:
        st.write("### SHAP waterfall plot - MQ")
        plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=mq_shap_values,
                base_values=MQ_explainer_model.expected_value[0],
                data=original_input[0],
                feature_names=features
            ),
            max_display=15,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.close()

    with col5:
        st.write("### SHAP waterfall plot - MS")
        plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=ms_shap_values,
                base_values=MS_explainer_model.expected_value[0],
                data=original_input[0],
                feature_names=features
            ),
            max_display=15,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.close()

    with col6:
        st.write("### SHAP waterfall plot - FV")
        plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=fv_shap_values,
                base_values=FV_explainer_model.expected_value[0],
                data=original_input[0],
                feature_names=features
            ),
            max_display=15,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.close()

# 定义每个组分的成本、碳足迹和隐含能参数
inventory_data = {
    'Fiber': {
        'Glass fiber': {'cost': 0.63, 'carbon_footprint': 2.04, 'energy_consumption': 48.33},
        'Mineral fiber': {'cost': 3.25, 'carbon_footprint': 0.06, 'energy_consumption': 0.96},
        'Plastic fiber': {'cost': 6.06, 'carbon_footprint': 3.02, 'energy_consumption': 94.5},
        'Steel fiber': {'cost': 6.00, 'carbon_footprint': 2.47, 'energy_consumption': 28.28},
        'Carbon fiber': {'cost': 24.55, 'carbon_footprint': 22.06, 'energy_consumption': 307.4},
        'Bio-fiber': {'cost': 0.60, 'carbon_footprint': 0.87, 'energy_consumption': 25.0}
    },
    'Asphalt': {'cost': 0.77, 'carbon_footprint': 0.29, 'energy_consumption': 4.17},
    'Coarse aggregate': {'cost': 0.043, 'carbon_footprint': 0.0094, 'energy_consumption': 0.068},
    'Fine aggregate': {'cost': 0.027, 'carbon_footprint': 0.014, 'energy_consumption': 0.13},
    'Filler': {'cost': 0.084, 'carbon_footprint': 0.58, 'energy_consumption': 3.08}
}


def get_fiber_params(fiber_type):
    fiber_mapping = {
        1: 'Glass fiber',
        2: 'Mineral fiber',
        3: 'Plastic fiber',
        4: 'Steel fiber',
        5: 'Carbon fiber',
        6: 'Bio-fiber'
    }
    fiber_name = fiber_mapping.get(fiber_type, None)
    if fiber_name:
        return inventory_data['Fiber'][fiber_name]
    else:
        return None


def calculate_lca_components(AC, FC, Ag2_36, FT):
    # 常数
    coarse_agg_density = 2.74  # Coarse aggregate density (g/cm³)
    fine_agg_density = 2.68  # Fine aggregate density (g/cm³)
    filler_density = 2.81  # Filler aggregate density (g/cm³)
    filler_ratio = 0.062  # Ratio of filler to aggregate

    # 计算 Aggregate + Filler content
    agg_filler_content_wei = 100 - AC - FC

    # 计算 Filler content (vol.%)
    filler_content_vol = agg_filler_content_wei * filler_ratio

    # 计算 Fine content (vol.%)
    fine_content_vol = Ag2_36 - filler_content_vol

    # 计算 Coarse content (vol.%)
    coarse_content_vol = 100 - AC - FC - filler_content_vol - fine_content_vol

    # 计算 Aggregate + Filler weight
    agg_filler_weight = (
            coarse_content_vol * coarse_agg_density +
            fine_content_vol * fine_agg_density +
            filler_content_vol * filler_density
    )

    # 计算 Coarse content
    coarse_content_wei = (
            (coarse_content_vol * coarse_agg_density / agg_filler_weight) *
            ((coarse_content_vol + fine_content_vol + filler_content_vol) / 100) * 100
    )

    # 计算 Fine content
    fine_content_wei = (
            (fine_content_vol * fine_agg_density / agg_filler_weight) *
            ((coarse_content_vol + fine_content_vol + filler_content_vol) / 100) * 100
    )

    # 计算 Filler content
    filler_content_wei = (
            (filler_content_vol * filler_density / agg_filler_weight) *
            ((coarse_content_vol + fine_content_vol + filler_content_vol) / 100) * 100
    )

    # 获取纤维参数
    fiber_params = get_fiber_params(FT)

    # 初始化总成本、碳足迹和隐含能
    total_cost = 0
    total_carbon_footprint = 0
    total_energy_consumption = 0

    # 计算 Asphalt 的贡献
    asphalt_params = inventory_data['Asphalt']
    total_cost += AC / 100 * asphalt_params['cost']
    total_carbon_footprint += AC / 100 * asphalt_params['carbon_footprint']
    total_energy_consumption += AC / 100 * asphalt_params['energy_consumption']

    # 计算 Fiber 的贡献
    if fiber_params:
        total_cost += FC / 100 * fiber_params['cost']
        total_carbon_footprint += FC / 100 * fiber_params['carbon_footprint']
        total_energy_consumption += FC / 100 * fiber_params['energy_consumption']

    # 计算 Coarse aggregate 的贡献
    coarse_agg_params = inventory_data['Coarse aggregate']
    total_cost += coarse_content_wei / 100 * coarse_agg_params['cost']
    total_carbon_footprint += coarse_content_wei / 100 * coarse_agg_params['carbon_footprint']
    total_energy_consumption += coarse_content_wei / 100 * coarse_agg_params['energy_consumption']

    # 计算 Fine aggregate 的贡献
    fine_agg_params = inventory_data['Fine aggregate']
    total_cost += fine_content_wei / 100 * fine_agg_params['cost']
    total_carbon_footprint += fine_content_wei / 100 * fine_agg_params['carbon_footprint']
    total_energy_consumption += fine_content_wei / 100 * fine_agg_params['energy_consumption']

    # 计算 Filler 的贡献
    filler_params = inventory_data['Filler']
    total_cost += filler_content_wei / 100 * filler_params['cost']
    total_carbon_footprint += filler_content_wei / 100 * filler_params['carbon_footprint']
    total_energy_consumption += filler_content_wei / 100 * filler_params['energy_consumption']

    return {
        'AC': AC,
        'FC': FC,
        'Coarse content': coarse_content_wei,
        'Fine content': fine_content_wei,
        'Filler content': filler_content_wei,
        'Total cost': total_cost,
        'Total carbon footprint': total_carbon_footprint,
        'Total energy consumption': total_energy_consumption
    }


def plot_contribution_pie_chart(contributions, labels, title, title_fontsize=22):  # 新增参数
    fig, ax = plt.subplots(figsize=(6, 6))

    wedges, texts, autotexts = ax.pie(
        contributions,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.3),
        textprops=dict(color="black")
    )

    ax.set_title(title, fontsize=title_fontsize)  # 使用传入的字体大小

    return fig


def styled_metric(label, value, label_fontsize="25px", value_fontsize="35px"):
    st.markdown(f"""
    <div style="font-size: {label_fontsize}; font-weight: bold; margin-bottom: 8px;">
        {label}<br>
        <span style="font-size: {value_fontsize}; font-weight: normal;">{value}</span>
    </div>
    """, unsafe_allow_html=True)


# LCA结果
st.markdown(
    "<div style='background-color:orange;padding:2px 10px;border-radius:5px;display:inline-block;margin-top:20px;margin-bottom:20px;'>"
    "<h2 style='color:white;margin:0;font-size:30px;text-align:left;'>Life cycle assessment</h2>"
    "</div>",
    unsafe_allow_html=True
)

if 'mq_prediction' in st.session_state:
    original_input = np.array(st.session_state['input_data'])[0]  # 获取原始输入数据

    # 提取需要的参数
    AC = original_input[3]
    FC = original_input[11]
    Ag2_36 = original_input[7]
    FT_value = int(original_input[10])
    mq_value = st.session_state['mq_prediction']

    # 调用计算函数
    lca_result = calculate_lca_components(AC, FC, Ag2_36, FT_value)

    # 计算 MQ-normalized 的各项指标
    mq_normalized_cost = lca_result['Total cost'] / mq_value
    mq_normalized_carbon_footprint = lca_result['Total carbon footprint'] / mq_value
    mq_normalized_energy_consumption = lca_result['Total energy consumption'] / mq_value

    # 显示结果
    st.header("Sample composition")
    col1, col2, col3 = st.columns(3)
    with col1:
        styled_metric("Asphalt content", f"{lca_result['AC']:.2f} %")
        styled_metric("Coarse aggregate content", f"{lca_result['Coarse content']:.2f} %")

    with col2:
        styled_metric("Fiber content", f"{lca_result['FC']:.2f} %")
        styled_metric("Fine aggregate content", f"{lca_result['Fine content']:.2f} %")

    with col3:
        styled_metric("Fiber type", fiber_type_mapping.get(FT_value, "Unknown"))
        styled_metric("Filler content", f"{lca_result['Filler content']:.2f} %")

    # 显示 Total 和 MQ-normalized 的结果
    st.header("LCA results")
    col4, col5, col6 = st.columns(3)
    with col4:
        styled_metric("Total cost", f"{lca_result['Total cost']:.2f} USD/Kg")
        styled_metric("MQ-normalized cost", f"{mq_normalized_cost:.2f} USD/(kN/mm)")

    with col5:
        styled_metric("Total carbon footprint", f"{lca_result['Total carbon footprint']:.2f} kg/Kg")
        styled_metric("MQ-normalized carbon footprint", f"{mq_normalized_carbon_footprint:.2f} kg/(kN/mm)")

    with col6:
        styled_metric("Total energy consumption", f"{lca_result['Total energy consumption']:.2f} MJ/Kg")
        styled_metric("MQ-normalized energy consumption",
                      f"{mq_normalized_energy_consumption:.2f} MJ/(kN/mm)")

    # 定义各项贡献值和标签
    contribution_labels = ['Asphalt', 'Fiber', 'Coarse aggregate', 'Fine aggregate', 'Filler']

    # 总成本贡献值
    cost_contributions = [
        lca_result['AC'] * inventory_data['Asphalt']['cost'] / 100,
        lca_result['FC'] * get_fiber_params(FT_value)['cost'] / 100 if FT_value else 0,
        lca_result['Coarse content'] * inventory_data['Coarse aggregate']['cost'] / 100,
        lca_result['Fine content'] * inventory_data['Fine aggregate']['cost'] / 100,
        lca_result['Filler content'] * inventory_data['Filler']['cost'] / 100
    ]

    # 碳足迹贡献值
    carbon_footprint_contributions = [
        lca_result['AC'] * inventory_data['Asphalt']['carbon_footprint'] / 100,
        lca_result['FC'] * get_fiber_params(FT_value)['carbon_footprint'] / 100 if FT_value else 0,
        lca_result['Coarse content'] * inventory_data['Coarse aggregate']['carbon_footprint'] / 100,
        lca_result['Fine content'] * inventory_data['Fine aggregate']['carbon_footprint'] / 100,
        lca_result['Filler content'] * inventory_data['Filler']['carbon_footprint'] / 100
    ]

    # 隐含能贡献值
    energy_consumption_contributions = [
        lca_result['AC'] * inventory_data['Asphalt']['energy_consumption'] / 100,
        lca_result['FC'] * get_fiber_params(FT_value)['energy_consumption'] / 100 if FT_value else 0,
        lca_result['Coarse content'] * inventory_data['Coarse aggregate']['energy_consumption'] / 100,
        lca_result['Fine content'] * inventory_data['Fine aggregate']['energy_consumption'] / 100,
        lca_result['Filler content'] * inventory_data['Filler']['energy_consumption'] / 100
    ]

    # 绘制并展示饼状图
    st.header("Contribution pie charts")
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })

    col7, col8, col9 = st.columns(3)
    with col7:
        cost_fig = plot_contribution_pie_chart(cost_contributions, contribution_labels, "Cost contribution")
        st.pyplot(cost_fig)

    with col8:
        carbon_footprint_fig = plot_contribution_pie_chart(carbon_footprint_contributions, contribution_labels,
                                                           "Carbon footprint contribution")
        st.pyplot(carbon_footprint_fig)

    with col9:
        energy_consumption_fig = plot_contribution_pie_chart(energy_consumption_contributions, contribution_labels,
                                                             "Energy consumption contribution")
        st.pyplot(energy_consumption_fig)
    status_placeholder.success("Calculation completed ✅")
