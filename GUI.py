import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import dill

# 显示中文和负号
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
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
# 创建三列布局
col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("Asphalt properties"):
        st.markdown("""
        <style>
        .stNumberInput label {
            font-size: 20px;
            font-weight: bold;
        }
        .stNumberInput input {
            font-size: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

        AC = st.number_input("Asphalt content (AC) [wei.%]", 2.0, 15.0, 5.0, step=0.1)
        Pe = st.number_input("Penetration (Pe) [0.1 mm]", 30.0, 110.0, 80.0, step=0.1)
        Du = st.number_input("Ductility (Du) [cm]", 20.0, 200.0, 100.0, step=1.0)
        SP = st.number_input("Softening point (SP) [℃]", 40.0, 60.0, 50.0, step=0.1)

with col2:
    with st.expander("Aggregate properties"):
        Ag2_36 = st.number_input("Aggregate passing 2.36mm (Ag2.36) [%]", 0.0, 100.0, 35.0, step=0.01, )
        Ag4_75 = st.number_input("Aggregate passing 4.75mm (Ag4.75) [%]", 0.0, 100.0, 45.0, step=0.01)
        Ag9_5 = st.number_input("Aggregate passing 9.5mm (Ag9.5) [%]", 0.0, 100.0, 70.0, step=0.01)
        AV = st.number_input("Air voids (AV) [%]", 0.0, 25.0, 4.0, step=0.01)
        VMA = st.number_input("Voids in mineral aggregate (VMA) [%]", 5.0, 70.0, 15.0, step=0.01)
        VFA = st.number_input("Voids filled with asphalt (VFA) [%]", 10.0, 100.0, 75.0, step=0.01)

with col3:
    with st.expander("Fiber properties"):
        FC = st.number_input("Fiber content (FC) [%]", 0.0, 7.0, 0.3, step=0.01)
        FT = st.selectbox(
            "Fiber type (FT)",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: fiber_type_mapping[x],  # 显示纤维类型名称
            index=0,
            help="Choose the type of fiber."
        )
        FL = st.number_input("Fiber length (FL) [mm]", 0.0, 130.0, 6.0, step=0.01)
        TS = st.number_input("Tensile strength (TS) [MPa]", 0.0, 10000.0, 50.0, step=0.1)
        MT = st.number_input("Melting temperature (MT) [℃]", 50.0, 5000.0, 150.0, step=1.0)

# 构建输入数据
input_data = [[Pe, Du, SP, AC, AV, VMA, VFA, Ag2_36, Ag4_75, Ag9_5,
               FT, FC, FL, TS, MT]]

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

    # 更新状态为完成

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
        st.markdown("<p style='font-size:20px; font-weight:bold;'>Marshall quotient (MQ)</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;text-align:center;'>"
            f"<span style='font-size:18px; font-weight:bold;'>{st.session_state['mq_prediction']:.2f} kN/mm</span></div>",
            unsafe_allow_html=True
        )

    with col_ms:
        st.markdown("<p style='font-size:20px; font-weight:bold;'>Marshall stability (MS)</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;text-align:center;'>"
            f"<span style='font-size:18px; font-weight:bold;'>{st.session_state['ms_prediction']:.2f} kN</span></div>",
            unsafe_allow_html=True
        )

    with col_fv:
        st.markdown("<p style='font-size:20px; font-weight:bold;'>Marshall flow value (FV)</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;text-align:center;'>"
            f"<span style='font-size:18px; font-weight:bold;'>{st.session_state['fv_prediction']:.2f} mm</span></div>",
            unsafe_allow_html=True
        )

    # 显示输入参数汇总
    st.subheader("Input parameters summary")
    input_df = pd.DataFrame(st.session_state['input_data'], columns=features)
    st.dataframe(input_df.style.format("{:.2f}"), use_container_width=True)

else:
    st.info("Please input parameters and click Predict three objectives' to see results.")

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
        st.write("#### SHAP force plot - MQ")
        mq_shap_values = MQ_explainer_model.shap_values(original_input)[0]

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
        st.write("#### SHAP force plot - MS")
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
        st.write("#### SHAP force plot - FV")
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
        st.write("#### SHAP waterfall plot - MQ")
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
        st.write("#### SHAP waterfall plot - MS")
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
        st.write("#### SHAP waterfall plot - FV")
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

# ... existing code ...

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

    # 计算 Aggregate + Filler content (wei.%)
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

    # 计算 Coarse content (wei.%)
    coarse_content_wei = (
            (coarse_content_vol * coarse_agg_density / agg_filler_weight) *
            ((coarse_content_vol + fine_content_vol + filler_content_vol) / 100) * 100
    )

    # 计算 Fine content (wei.%)
    fine_content_wei = (
            (fine_content_vol * fine_agg_density / agg_filler_weight) *
            ((coarse_content_vol + fine_content_vol + filler_content_vol) / 100) * 100
    )

    # 计算 Filler content (wei.%)
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


def plot_contribution_pie_chart(contributions, labels, title):
    """
    绘制贡献饼状图

    :param contributions: list of float, 各项贡献值
    :param labels: list of str, 各项标签
    :param title: str, 图表标题
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制饼状图
    wedges, texts, autotexts = ax.pie(
        contributions,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.3),
        textprops=dict(color="black")
    )

    # 设置标题
    ax.set_title(title, fontsize=16)

    # 返回图表对象
    return fig


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
        st.metric("Asphalt content (wei.%)", f"{lca_result['AC']:.2f} %")
        st.metric("Coarse aggregate content (wei.%)", f"{lca_result['Coarse content']:.2f} %")

    with col2:
        st.metric("Fiber content (wei.%)", f"{lca_result['FC']:.2f} %")
        st.metric("Fine aggregate content (wei.%)", f"{lca_result['Fine content']:.2f} %")

    with col3:
        st.metric("Fiber type", fiber_type_mapping.get(FT_value, "Unknown"))
        st.metric("Filler content (wei.%)", f"{lca_result['Filler content']:.2f} %")

    # 显示 Total 和 MQ-normalized 的结果
    st.header("LCA results")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Total cost (USD/Kg)", f"{lca_result['Total cost']:.2f} USD/Kg")
        st.metric("MQ-normalized cost (USD/(kN/mm))", f"{mq_normalized_cost:.2f} USD/(kN/mm)")

    with col5:
        st.metric("Total carbon footprint (kg/Kg)", f"{lca_result['Total carbon footprint']:.2f} kg/Kg")
        st.metric("MQ-normalized carbon footprint (kg/(kN/mm))", f"{mq_normalized_carbon_footprint:.2f} kg/(kN/mm)")

    with col6:
        st.metric("Total energy consumption (MJ/Kg)", f"{lca_result['Total energy consumption']:.2f} MJ/Kg")
        st.metric("MQ-normalized energy consumption (MJ/(kN/mm))", f"{mq_normalized_energy_consumption:.2f} MJ/(kN/mm)")

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
