import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib as mpl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
FIGURE_DIR = PROJECT_ROOT / 'outputs' / 'figures'

# 读取数据文件
electric_csv = pd.read_csv(RAW_DATA_DIR / "electric_load.csv", encoding='utf-8')
residential_heat_csv = pd.read_csv(RAW_DATA_DIR / "residential_heating.csv", encoding='utf-8')
commercial_heat_csv = pd.read_csv(RAW_DATA_DIR / "commercial_heating.csv", encoding='utf-8')
industry_1_csv = pd.read_csv(RAW_DATA_DIR / "chemicals_and_petrochemicals.csv", encoding='utf-8')
industry_2_csv = pd.read_csv(RAW_DATA_DIR / "food_and_tobacco.csv", encoding='utf-8')
industry_3_csv = pd.read_csv(RAW_DATA_DIR / "iron_and_steel.csv", encoding='utf-8')
industry_4_csv = pd.read_csv(RAW_DATA_DIR / "metalic_minerals.csv", encoding='utf-8')
industry_5_csv = pd.read_csv(RAW_DATA_DIR / "paper.csv", encoding='utf-8')
cooling_csv = pd.read_csv(RAW_DATA_DIR / "cooling_commercial.csv", encoding='utf-8')
solar_csv = pd.read_csv(RAW_DATA_DIR / "solar.csv", encoding='utf-8')

#提取数据
electric_load = electric_csv['value'].tolist()
residential_heating = residential_heat_csv['load'].tolist()
commercial_heating = commercial_heat_csv['load'].tolist()
industry_heating_1 = industry_1_csv['load'].tolist()
industry_heating_2 = industry_2_csv['load'].tolist()
industry_heating_3 = industry_3_csv['load'].tolist()
industry_heating_4 = industry_4_csv['load'].tolist()
industry_heating_5 = industry_5_csv['load'].tolist()
cooling_load = cooling_csv['load'].tolist()
solar = solar_csv['value'].tolist()

#设置负荷
#电负荷
commercial_e = electric_load[:8760]
industry_e = electric_load[32855:41615]
residential_e = electric_load[67175:75935]
residential_e = [i / 5 for i in residential_e]
commercial_e= [i / 5 for i in commercial_e]
industry_e = [i / 5 for i in industry_e]
#热负荷
residential_h = residential_heating[:8760]
commercial_h = commercial_heating[:8760]
industry_h = []
for i in range(8760):
    industry_h.append(industry_heating_1[:8760][i]+industry_heating_2[:8760][i]+industry_heating_3[:8760][i]+
                      industry_heating_4[:8760][i]+industry_heating_5[:8760][i])
residential_h = [i * 1.5 for i in residential_h]
commercial_h = [i * 3 for i in commercial_h]
industry_h = [i * 3 for i in industry_h]
#冷负荷
residential_c = cooling_load[:8760]
commercial_c = cooling_load[8760:17520]
industry_c = cooling_load[17520:26280]
commercial_c = [i * 1.5 for i in commercial_c]

residential_solar = solar[:8760]
commercial_solar = solar[8760:17520]
industry_solar = solar[17544:26304]

#买电价格
electricity_price = [0.4] * 7 + [0.6] * 4 + [1.1] * 3 + [0.6] * 4 + [1.1] * 4 + [0.4] * 2
electricity_price = electricity_price * 365

residential_electric_load = np.array(residential_e).reshape((len(residential_e),1))
commercial_electric_load = np.array(commercial_e).reshape((len(commercial_e),1))
industry_electric_load = np.array(industry_e).reshape((len(industry_e),1))

residential_heat_load = np.array(residential_h).reshape((len(residential_h),1))
commercial_heat_load = np.array(commercial_h).reshape((len(commercial_h),1))
industry_heat_load = np.array(industry_h).reshape((len(industry_h),1))

residential_cooling_load = np.array(residential_c).reshape((len(residential_h),1))
commercial_cooling_load = np.array(commercial_c).reshape((len(commercial_h),1))
industry_cooling_load = np.array(industry_c).reshape((len(industry_h),1))

residential_PV = np.array(residential_solar).reshape((len(residential_solar),1))
commercial_PV = np.array(commercial_solar).reshape((len(commercial_solar),1))
industry_PV = np.array(industry_solar).reshape((len(industry_solar),1))

TOU = np.array(electricity_price).reshape((len(electricity_price)),1)

residential_data_concat = np.concatenate((residential_electric_load,residential_heat_load,residential_cooling_load,TOU,residential_PV),axis=-1)
commercial_data_concat = np.concatenate((commercial_electric_load,commercial_heat_load,commercial_cooling_load,TOU,commercial_PV),axis=-1)
industry_data_concat = np.concatenate((industry_electric_load,industry_heat_load,industry_cooling_load,TOU,industry_PV),axis=-1)


def extract(data):
    """
    直接提取并拼接所有月份前15天数据
    """
    # 初始化空列表用于收集数据
    all_first_10_days = []

    # 每个月的小时数（平年）
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    hour_index = 0

    days_to_extract = 5 #取每月的前多少天
    hours_to_extract = days_to_extract * 24

    for days_in_month in month_days:
        # 提取数据
        month_first_10 = data[hour_index:hour_index + hours_to_extract]
        all_first_10_days.append(month_first_10)

        # 移动到下个月
        hour_index += days_in_month * 24

    # 将所有月份数据垂直堆叠成一个数组
    result = np.vstack(all_first_10_days)

    return result


def extract_seasonal_test_days(data):
    """Extract Jan 21, Apr 21, Jul 21, and Oct 21, 24 hours each."""
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    test_months = [0, 3, 6, 9]
    test_day_index = 20
    slices = []
    for month in test_months:
        start = (sum(month_days[:month]) + test_day_index) * 24
        slices.append(data[start:start + 24])
    return np.vstack(slices)


if __name__ == '__main__':
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 取数据集中各类数据的最小值
    min_data_residential = np.min(residential_data_concat, axis=0)
    min_data_commercial = np.min(commercial_data_concat, axis=0)
    min_data_industry = np.min(industry_data_concat, axis=0)

    # 取数据集中各类数据的最大值
    max_data_residential = np.max(residential_data_concat, axis=0)
    max_data_commercial = np.max(commercial_data_concat, axis=0)
    max_data_industry = np.max(industry_data_concat, axis=0)

    # 对数据进行归一化
    full_data_residential = (residential_data_concat - min_data_residential) / (max_data_residential - min_data_residential)
    full_data_commercial = (commercial_data_concat - min_data_commercial) / (max_data_commercial - min_data_commercial)
    full_data_industry = (industry_data_concat - min_data_industry) / (max_data_industry - min_data_industry)

    # 取训练数据
    data_residential = extract(full_data_residential)
    data_commercial = extract(full_data_commercial)
    data_industry = extract(full_data_industry)

    # 取测试数据：四季代表日，每季 24 小时，共 96 小时。
    test_data_residential_concat = extract_seasonal_test_days(full_data_residential)
    test_data_commercial_concat = extract_seasonal_test_days(full_data_commercial)
    test_data_industry_concat = extract_seasonal_test_days(full_data_industry)

    np.save(PROCESSED_DATA_DIR / 'data_residential', data_residential)
    np.save(PROCESSED_DATA_DIR / 'data_commercial', data_commercial)
    np.save(PROCESSED_DATA_DIR / 'data_industry', data_industry)
    np.save(PROCESSED_DATA_DIR / 'min_data_residential', min_data_residential)
    np.save(PROCESSED_DATA_DIR / 'min_data_commercial', min_data_commercial)
    np.save(PROCESSED_DATA_DIR / 'min_data_industry', min_data_industry)
    np.save(PROCESSED_DATA_DIR / 'max_data_residential', max_data_residential)
    np.save(PROCESSED_DATA_DIR / 'max_data_commercial', max_data_commercial)
    np.save(PROCESSED_DATA_DIR / 'max_data_industry', max_data_industry)

    np.save(PROCESSED_DATA_DIR / 'test_data_residential',test_data_residential_concat)
    np.save(PROCESSED_DATA_DIR / 'test_data_commercial',test_data_commercial_concat)
    np.save(PROCESSED_DATA_DIR / 'test_data_industry',test_data_industry_concat)


    # time = list(range(8760))
    # plt.style.use('seaborn-v0_8-whitegrid')
    # mpl.rcParams.update({
    #     'font.family': 'Times New Roman',
    #     'font.size': 10,
    #     'axes.titlesize': 12,
    #     'axes.labelsize': 10,
    #     'xtick.labelsize': 9,
    #     'ytick.labelsize': 9,
    #     'legend.fontsize': 9,
    #     'figure.dpi': 300,
    #     'savefig.dpi': 300,
    #     'axes.grid': True,
    #     'grid.alpha': 0.3,
    #     'grid.linestyle': '--',
    #     'axes.linewidth': 0.8,
    #     'figure.autolayout': True,
    #     'legend.frameon': True,
    #     'legend.framealpha': 0.8
    # })

    # plt.figure(figsize=(6, 3), dpi=300)
    # plt.plot(time, residential_electric_load, label='residential_e', color='blue', linestyle='-', linewidth=1)
    # plt.plot(time, commercial_electric_load, label='commercial_e', color='red', linestyle='-', linewidth=1)
    # plt.plot(time, industry_electric_load, label='industry_e', color='green', linestyle='-', linewidth=1)
    # plt.xlabel('Time (h)', fontsize=14)
    # plt.ylabel('Power (kW)', fontsize=14)
    # plt.legend(fontsize=10, frameon=True, framealpha=1, edgecolor='black', loc='best')
    # plt.savefig(FIGURE_DIR / 'electric_load')
    #
    # plt.figure(figsize=(6, 3), dpi=300)
    # plt.plot(time, residential_cooling_load, label='residential_c', color='blue', linestyle='-', linewidth=1)
    # plt.plot(time, commercial_cooling_load, label='commercial_c', color='red', linestyle='-', linewidth=1)
    # plt.plot(time, industry_cooling_load, label='industry_c', color='green', linestyle='-', linewidth=1)
    # plt.xlabel('Time (h)', fontsize=14)
    # plt.ylabel('Power (kW)', fontsize=14)
    # plt.legend(fontsize=10, frameon=True, framealpha=1, edgecolor='black', loc='best')
    # plt.savefig(FIGURE_DIR / 'cooling_load')
    #
    # plt.figure(figsize=(6, 3), dpi=300)
    # plt.plot(time, residential_heat_load, label='residential_h', color='blue', linestyle='-', linewidth=1)
    # plt.plot(time, commercial_heat_load, label='commercial_h', color='red', linestyle='-', linewidth=1)
    # plt.plot(time, industry_heat_load, label='industry_h', color='green', linestyle='-', linewidth=1)
    # plt.xlabel('Time (h)', fontsize=14)
    # plt.ylabel('Power (kW)', fontsize=14)
    # plt.legend(fontsize=10, frameon=True, framealpha=1, edgecolor='black', loc='best')
    # plt.savefig(FIGURE_DIR / 'heat_load')
    #
    # plt.figure(figsize=(6, 3), dpi=300)
    # plt.plot(time, residential_solar, label='residential_solar', color='blue', linestyle='-', linewidth=1)
    # plt.plot(time, commercial_solar, label='commercial_solar', color='red', linestyle='-', linewidth=1)
    # plt.plot(time, industry_solar, label='industry_solar', color='green', linestyle='-', linewidth=1)
    # plt.xlabel('Time (h)', fontsize=14)
    # plt.ylabel('Power (kW)', fontsize=14)
    # plt.legend(fontsize=10, frameon=True, framealpha=1, edgecolor='black', loc='best')
    # plt.savefig(FIGURE_DIR / 'solar')
