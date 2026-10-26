from src.envs.equipment import GB, CHP, EB, ESS, TES, HCHP, AC, CER
import numpy as np
from src.envs.market import (
    CDA,
    carbon_aware_CDA,
    carbon_dividend_settlement,
    profit,
    volume_preserving_carbon_aware_CDA,
)
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVED_ARRAY_DIR = PROJECT_ROOT / 'data' / 'processed'

state_train_residential = np.load(SAVED_ARRAY_DIR / 'data_residential.npy')
state_train_commercial = np.load(SAVED_ARRAY_DIR / 'data_commercial.npy')
state_train_industry = np.load(SAVED_ARRAY_DIR / 'data_industry.npy')
state_test_residential = np.load(SAVED_ARRAY_DIR / 'test_data_residential.npy')
state_test_commercial = np.load(SAVED_ARRAY_DIR / 'test_data_commercial.npy')
state_test_industry = np.load(SAVED_ARRAY_DIR / 'test_data_industry.npy')
min_residential = np.load(SAVED_ARRAY_DIR / 'min_data_residential.npy')
min_commercial = np.load(SAVED_ARRAY_DIR / 'min_data_commercial.npy')
min_industry = np.load(SAVED_ARRAY_DIR / 'min_data_industry.npy')
max_residential = np.load(SAVED_ARRAY_DIR / 'max_data_residential.npy')
max_commercial = np.load(SAVED_ARRAY_DIR / 'max_data_commercial.npy')
max_industry = np.load(SAVED_ARRAY_DIR / 'max_data_industry.npy')

class env():
    def __init__(
        self,
        market_mechanism="standard",
        cda_carbon_price=0.05,
        cda_dynamic_carbon_price=False,
        cda_carbon_pressure_scale=1.0,
        cda_carbon_intensity_ref=1.0,
        cda_carbon_price_min_multiplier=0.5,
        cda_carbon_price_max_multiplier=3.0,
    ):
        """
        居民区状态空间：电负荷(0)，热负荷(1)，冷负荷(2)，TOU(3)，光伏发电量(4)，电池电量(5)，储热量(6)
        居民区动作空间：氢热电联产电功率(0)，热电联产电功率(1)，电制冷机(2)，电池放电(3)，电交易价格(4)

        商业区状态空间：电负荷(0)，热负荷(1)，冷负荷(2)，TOU(3)，光伏发电量(4)，电池电量(5)，储热量(6)
        商业区动作空间：氢热电联产电功率(0)，热电联产电功率(1)，电制冷机(2),电池放电(3)，电交易价格(4)

        工业区状态空间：电负荷(0)，热负荷(1)，冷负荷(2)，TOU(3)，光伏发电量(4)，电池电量(5)，储热量(6)
        工业区动作空间：氢热电联产电功率(0)，热电联产电功率(1)，电池放电(2)，电交易价格(3)
        """

        self.residential_CHP = CHP(CHP_MAX=1000)
        self.commercial_CHP = CHP(CHP_MAX=1500)
        self.industry_CHP = CHP(CHP_MAX=2000)

        self.residential_HCHP = HCHP(HCHP_MAX=1000)
        self.commercial_HCHP = HCHP(HCHP_MAX=1500)
        self.industry_HCHP = HCHP(HCHP_MAX=2000)

        self.residential_EB = EB(EB_MAX=1000)
        self.commercial_EB = EB(EB_MAX=1000)

        self.industry_GB = GB(GB_MAX=1000)

        self.residential_CER = CER(CER_MAX=1000)
        self.commercial_CER = CER(CER_MAX=1000)

        self.residential_AC = AC(AC_MAX=1000)
        self.commercial_AC = AC(AC_MAX=1000)
        self.industry_AC = AC(AC_MAX=500)

        self.residential_ESS = ESS(max_storage=1000, min_storage=100, charge_max=500)
        self.commercial_ESS = ESS(max_storage=1000, min_storage=100, charge_max=500)
        self.industry_ESS = ESS(max_storage=2000, min_storage=200, charge_max=1000)

        self.residential_TES = TES(max_storage=1000, charge_max=500)
        self.commercial_TES = TES(max_storage=1000, charge_max=500)
        self.industry_TES = TES(max_storage=2000, charge_max=1000)

        self.gas_price = 3.2  # 天然气价格为3.2元/m3
        self.H2_price = 2.6  # 氢气价格为2.6元/m3
        self.EF = 0.5839  # 电网二氧化碳排放因子
        self.OF = 0.99  # 天然气碳氧化率
        self.NCV_gas = 35.6  # 天然气低位发热量(MJ/m3)
        self.CC = 15.3e-3  # 单位热值含碳量(kg/MJ)
        self.ratio_carbon_oxygen = 44 / 12  # C到CO2的换算系数

        self.market_mechanism = market_mechanism
        self.cda_carbon_price = cda_carbon_price
        self.cda_dynamic_carbon_price = cda_dynamic_carbon_price
        self.cda_carbon_pressure_scale = cda_carbon_pressure_scale
        self.cda_carbon_intensity_ref = max(cda_carbon_intensity_ref, 1e-9)
        self.cda_carbon_price_min_multiplier = cda_carbon_price_min_multiplier
        self.cda_carbon_price_max_multiplier = cda_carbon_price_max_multiplier
        self.current_cda_carbon_price = cda_carbon_price
        self.carbon_intensity_residential = 0.0
        self.carbon_intensity_commercial = 0.0
        self.carbon_intensity_industry = 0.0
        self.trade_record_e = []
        self.carbon_levy = {1: 0.0, 2: 0.0, 3: 0.0}
        self.carbon_credit = {1: 0.0, 2: 0.0, 3: 0.0}
        self.avoided_carbon_credit_basis = {1: 0.0, 2: 0.0, 3: 0.0}
        self.carbon_dividend_fund = 0.0
        self.carbon_dividend_balance_error = 0.0
        self.electricity_price = 0
        # 居民区参数
        self.state_dim_residential = 7
        self.action_dim_residential = 5
        self.P_HCHP_residential = 0
        self.P_CHP_residential = 0
        self.H_EB_residential = 0
        self.C_CER_residential = 0
        self.C_AC_residential = 0
        self.P_ESS_residential = 0
        self.H_TES_residential = 0
        self.electric_load_residential = 0
        self.heat_load_residential = 0
        self.cooling_load_residential = 0
        self.PV_residential = 0
        self.trade_e_residential = 0
        self.trade_h_residential = 0
        self.price_e_residential = 0
        self.price_h_residential = 0
        self.waste_h_residential = 0
        self.buy_from_grid_residential = 0
        self.sell_to_grid_residential = 0

        #商业区参数
        self.state_dim_commercial = 7
        self.action_dim_commercial = 5
        self.P_HCHP_commercial = 0
        self.P_CHP_commercial = 0
        self.H_EB_commercial = 0
        self.C_CER_commercial = 0
        self.C_AC_commercial = 0
        self.P_ESS_commercial = 0
        self.H_TES_commercial = 0
        self.electric_load_commercial = 0
        self.heat_load_commercial = 0
        self.cooling_load_commercial = 0
        self.PV_commercial = 0
        self.trade_e_commercial = 0
        self.trade_h_commercial = 0
        self.price_e_commercial = 0
        self.price_h_commercial = 0
        self.waste_h_commercial = 0
        self.buy_from_grid_commercial = 0
        self.sell_to_grid_commercial = 0

        # 工业区参数
        self.state_dim_industry = 7
        self.action_dim_industry = 4
        self.P_HCHP_industry = 0
        self.P_CHP_industry = 0
        self.H_GB_industry = 0
        self.C_AC_industry = 0
        self.P_ESS_industry = 0
        self.H_TES_industry = 0
        self.electric_load_industry = 0
        self.heat_load_industry = 0
        self.cooling_load_industry = 0
        self.PV_industry = 0
        self.trade_e_industry = 0
        self.trade_h_industry = 0
        self.price_e_industry = 0
        self.price_h_industry = 0
        self.waste_h_industry = 0
        self.buy_from_grid_industry = 0
        self.sell_to_grid_industry = 0

        self.n = 0

    def _max_tes_charge_heat(self, tes, tes_state):
        return max(
            0,
            (tes.max_storage - tes_state * tes.max_storage * (1 - tes.sigma_HS)) / tes.yita_ch
        )

    def _limit_cogeneration_heat(
        self,
        heat_sink,
        tes,
        tes_state,
        p_hchp,
        h_hchp,
        h2,
        p_chp,
        h_chp,
        gas_chp,
    ):
        max_heat_output = heat_sink + self._max_tes_charge_heat(tes, tes_state)
        current_heat_output = h_hchp + h_chp
        if current_heat_output <= max_heat_output or current_heat_output <= 0:
            return p_hchp, h_hchp, h2, p_chp, h_chp, gas_chp

        scale = max_heat_output / current_heat_output
        return (
            p_hchp * scale,
            h_hchp * scale,
            h2 * scale,
            p_chp * scale,
            h_chp * scale,
            gas_chp * scale,
        )

    def _electric_carbon_intensity(self, p_hchp, p_chp, p_ess, pv, gas_chp):
        direct_carbon = gas_chp * self.NCV_gas * self.CC * self.OF * self.ratio_carbon_oxygen
        electric_supply = max(p_hchp, 0.0) + max(p_chp, 0.0) + max(p_ess, 0.0) + max(pv, 0.0)
        if electric_supply <= 1e-9:
            return 0.0
        return direct_carbon / electric_supply

    def _cda_carbon_price(self, trades, carbon_intensities):
        if not self.cda_dynamic_carbon_price:
            return self.cda_carbon_price

        trades = np.array(trades, dtype=float)
        carbon_intensities = np.array(carbon_intensities, dtype=float)
        sell = np.maximum(trades, 0.0)
        buy = np.maximum(-trades, 0.0)
        sell_total = np.sum(sell)
        buy_total = np.sum(buy)
        if sell_total + buy_total <= 1e-9:
            return self.cda_carbon_price

        if sell_total > 1e-9:
            seller_carbon = np.sum(sell * np.maximum(carbon_intensities, 0.0)) / sell_total
        else:
            seller_carbon = np.mean(np.maximum(carbon_intensities, 0.0))

        carbon_pressure = seller_carbon / self.cda_carbon_intensity_ref
        market_pressure = buy_total / max(sell_total + buy_total, 1e-9)
        multiplier = 1.0 + self.cda_carbon_pressure_scale * (0.5 * carbon_pressure + 0.5 * market_pressure)
        multiplier = np.clip(
            multiplier,
            self.cda_carbon_price_min_multiplier,
            self.cda_carbon_price_max_multiplier,
        )
        return self.cda_carbon_price * multiplier

    def step(self, action, state, step, flag):
        state_copy = deepcopy(state)
        state_copy[0, :-2] = state_copy[0, :-2] * (max_residential - min_residential) + min_residential
        state_copy[1, :-2] = state_copy[1, :-2] * (max_commercial - min_commercial) + min_commercial
        state_copy[2, :-2] = state_copy[2, :-2] * (max_industry - min_industry) + min_industry

        self.electricity_price = state_copy[0,3]

        self.electric_load_residential = state_copy[0,0]
        self.heat_load_residential = state_copy[0,1]
        self.cooling_load_residential = state_copy[0,2]
        self.PV_residential = state_copy[0,4]

        self.electric_load_commercial = state_copy[1,0]
        self.heat_load_commercial = state_copy[1,1]
        self.cooling_load_commercial = state_copy[1,2]
        self.PV_commercial = state_copy[1,4]

        self.electric_load_industry = state_copy[2,0]
        self.heat_load_industry = state_copy[2,1]
        self.cooling_load_industry = state_copy[2,2]
        self.PV_industry = state_copy[2,4]

        #居民区动作
        a_HCHP_residential = (action[0,0] + 1) / 2
        self.P_HCHP_residential, H_HCHP_residential, H2_residential = self.residential_HCHP.HCHP(a_HCHP_residential)

        a_CHP_residential = (action[0,1] + 1) / 2
        self.P_CHP_residential, H_CHP_residential, gas_CHP_residential = self.residential_CHP.CHP(a_CHP_residential)

        a_CER_residential = (action[0,2] + 1) / 2
        self.C_CER_residential, P_CER_residential = self.residential_CER.CER(a_CER_residential)

        a_ESS_residential = action[0,3]
        self.P_ESS_residential = a_ESS_residential * self.residential_ESS.charge_max
        ESS_next_residential = self.residential_ESS.state_transmission(self.P_ESS_residential, abs(self.P_ESS_residential),
                                                                  state_copy[0,5])
        if ESS_next_residential > 1:
            P_ch = (self.residential_ESS.max_storage - state_copy[0,5] * self.residential_ESS.max_storage) / self.residential_ESS.yita_charge
            self.P_ESS_residential = -P_ch
            ESS_next_residential = 1
        elif ESS_next_residential < self.residential_ESS.min_state:
            P_dis = (state_copy[0,5] * self.residential_ESS.max_storage - self.residential_ESS.min_storage) * self.residential_ESS.yita_discharge
            self.P_ESS_residential = P_dis
            ESS_next_residential = self.residential_ESS.min_state

        #电交易价格
        self.price_e_residential = self.electricity_price / 2 + ((action[0,4] + 1) / 2) * (
                self.electricity_price / 2)

        #商业区动作
        a_HCHP_commercial = (action[1,0] + 1) / 2
        self.P_HCHP_commercial, H_HCHP_commercial, H2_commercial = self.commercial_HCHP.HCHP(a_HCHP_commercial)

        a_CHP_commercial = (action[1,1] + 1) / 2
        self.P_CHP_commercial, H_CHP_commercial, gas_CHP_commercial = self.commercial_CHP.CHP(a_CHP_commercial)

        a_CER_commercial = (action[1,2] + 1) / 2
        self.C_CER_commercial, P_CER_commercial = self.commercial_CER.CER(a_CER_commercial)

        a_ESS_commercial = action[1,3]
        self.P_ESS_commercial = a_ESS_commercial * self.commercial_ESS.charge_max
        ESS_next_commercial = self.commercial_ESS.state_transmission(self.P_ESS_commercial, abs(self.P_ESS_commercial),
                                                                  state_copy[1,5])
        if ESS_next_commercial > 1:
            P_ch = (self.commercial_ESS.max_storage - state_copy[1,5] * self.commercial_ESS.max_storage) / self.commercial_ESS.yita_charge
            self.P_ESS_commercial = -P_ch
            ESS_next_commercial = 1
        elif ESS_next_commercial < self.commercial_ESS.min_state:
            P_dis = (state_copy[1,5] * self.commercial_ESS.max_storage - self.commercial_ESS.min_storage) * self.commercial_ESS.yita_discharge
            self.P_ESS_commercial = P_dis
            ESS_next_commercial = self.commercial_ESS.min_state

        #电交易价格
        self.price_e_commercial = self.electricity_price / 2 + ((action[1,4] + 1) / 2) * (
                self.electricity_price / 2)

        #工业区动作
        a_HCHP_industry = (action[2,0] + 1) / 2
        self.P_HCHP_industry, H_HCHP_industry, H2_industry = self.industry_HCHP.HCHP(a_HCHP_industry)

        a_CHP_industry = (action[2,1] + 1) / 2
        self.P_CHP_industry, H_CHP_industry, gas_CHP_industry = self.industry_CHP.CHP(a_CHP_industry)

        a_ESS_industry = action[2,2]
        self.P_ESS_industry = a_ESS_industry * self.industry_ESS.charge_max
        ESS_next_industry = self.industry_ESS.state_transmission(self.P_ESS_industry, abs(self.P_ESS_industry),
                                                                  state_copy[2,5])
        if ESS_next_industry > 1:
            P_ch = (self.industry_ESS.max_storage - state_copy[2,5] * self.industry_ESS.max_storage) / self.industry_ESS.yita_charge
            self.P_ESS_industry = -P_ch
            ESS_next_industry = 1
        elif ESS_next_industry < self.industry_ESS.min_state:
            P_dis = (state_copy[2,5] * self.industry_ESS.max_storage - self.industry_ESS.min_storage) * self.industry_ESS.yita_discharge
            self.P_ESS_industry = P_dis
            ESS_next_industry = self.industry_ESS.min_state

        #电交易价格
        self.price_e_industry = self.electricity_price / 2 + ((action[2,3] + 1) / 2) * (
                self.electricity_price / 2)

        #计算设备冷出力
        if self.C_CER_residential > self.cooling_load_residential:
            self.C_CER_residential = self.cooling_load_residential
            P_CER_residential = self.C_CER_residential / self.residential_CER.COP_CER
        if self.C_CER_commercial > self.cooling_load_commercial:
            self.C_CER_commercial = self.cooling_load_commercial
            P_CER_commercial = self.C_CER_commercial / self.commercial_CER.COP_CER
        self.C_AC_industry = self.cooling_load_industry
        self.C_AC_residential = self.cooling_load_residential - self.C_CER_residential
        self.C_AC_commercial = self.cooling_load_commercial - self.C_CER_commercial
        H_AC_residential = self.C_AC_residential / self.residential_AC.COP_AC
        H_AC_commercial = self.C_AC_commercial / self.commercial_AC.COP_AC
        H_AC_industry = self.C_AC_industry / self.industry_AC.COP_AC

        self.P_HCHP_residential, H_HCHP_residential, H2_residential, \
            self.P_CHP_residential, H_CHP_residential, gas_CHP_residential = \
            self._limit_cogeneration_heat(
                self.heat_load_residential + H_AC_residential,
                self.residential_TES,
                state_copy[0, 6],
                self.P_HCHP_residential,
                H_HCHP_residential,
                H2_residential,
                self.P_CHP_residential,
                H_CHP_residential,
                gas_CHP_residential,
            )

        self.P_HCHP_commercial, H_HCHP_commercial, H2_commercial, \
            self.P_CHP_commercial, H_CHP_commercial, gas_CHP_commercial = \
            self._limit_cogeneration_heat(
                self.heat_load_commercial + H_AC_commercial,
                self.commercial_TES,
                state_copy[1, 6],
                self.P_HCHP_commercial,
                H_HCHP_commercial,
                H2_commercial,
                self.P_CHP_commercial,
                H_CHP_commercial,
                gas_CHP_commercial,
            )

        self.P_HCHP_industry, H_HCHP_industry, H2_industry, \
            self.P_CHP_industry, H_CHP_industry, gas_CHP_industry = \
            self._limit_cogeneration_heat(
                self.heat_load_industry + H_AC_industry,
                self.industry_TES,
                state_copy[2, 6],
                self.P_HCHP_industry,
                H_HCHP_industry,
                H2_industry,
                self.P_CHP_industry,
                H_CHP_industry,
                gas_CHP_industry,
            )

        #计算需要增发或者储存的热能
        H_residential = (self.heat_load_residential + H_AC_residential) - (H_HCHP_residential + H_CHP_residential)
        H_commercial = (self.heat_load_commercial + H_AC_commercial) - (H_HCHP_commercial + H_CHP_commercial)
        H_industry = (self.heat_load_industry + H_AC_industry) - (H_HCHP_industry + H_CHP_industry)

        self.H_EB_residential = 0
        self.waste_h_residential = 0
        self.H_TES_residential = H_residential
        TES_next_residential = self.residential_TES.state_transmission(self.H_TES_residential, abs(self.H_TES_residential), state_copy[0,6])
        if TES_next_residential > 1:
            H_ch = (self.residential_TES.max_storage - state_copy[0,6] * self.residential_TES.max_storage * (1 - self.residential_TES.sigma_HS)) / self.residential_TES.yita_ch
            self.waste_h_residential = abs(self.H_TES_residential) - H_ch
            self.H_TES_residential = -H_ch
            TES_next_residential = 1
        elif TES_next_residential <= self.residential_TES.min_state:
            if state_copy[0,6] * (1 - self.residential_TES.sigma_HS) <= self.residential_TES.min_state:
                H_dis = 0
            else:
                H_dis = (state_copy[0,6] * (1 - self.residential_TES.sigma_HS) * self.residential_TES.max_storage - self.residential_TES.min_storage) * self.residential_TES.yita_dis
            self.H_EB_residential = max(0, abs(self.H_TES_residential) - H_dis)
            self.H_TES_residential = H_dis
            TES_next_residential = self.residential_TES.min_state

        self.H_EB_commercial = 0
        self.waste_h_commercial = 0
        self.H_TES_commercial = H_commercial
        TES_next_commercial = self.commercial_TES.state_transmission(self.H_TES_commercial, abs(self.H_TES_commercial), state_copy[1,6])
        if TES_next_commercial > 1:
            H_ch = (self.commercial_TES.max_storage - state_copy[1,6] * self.commercial_TES.max_storage * (1 - self.commercial_TES.sigma_HS)) / self.commercial_TES.yita_ch
            self.waste_h_commercial = abs(self.H_TES_commercial) - H_ch
            self.H_TES_commercial = -H_ch
            TES_next_commercial = 1
        elif TES_next_commercial <= self.commercial_TES.min_state:
            if state_copy[1,6] * (1 - self.commercial_TES.sigma_HS) <= self.commercial_TES.min_state:
                H_dis = 0
            else:
                H_dis = (state_copy[1,6] * (1 - self.commercial_TES.sigma_HS) * self.commercial_TES.max_storage - self.commercial_TES.min_storage) * self.commercial_TES.yita_dis
            self.H_EB_commercial = max(0, abs(self.H_TES_commercial) - H_dis)
            self.H_TES_commercial = H_dis
            TES_next_commercial = self.commercial_TES.min_state

        self.H_GB_industry = 0
        self.waste_h_industry = 0
        self.H_TES_industry = H_industry
        TES_next_industry = self.industry_TES.state_transmission(self.H_TES_industry, abs(self.H_TES_industry), state_copy[2,6])
        if TES_next_industry > 1:
            H_ch = (self.industry_TES.max_storage - state_copy[2,6] * self.industry_TES.max_storage * (1 - self.industry_TES.sigma_HS)) / self.industry_TES.yita_ch
            self.waste_h_industry = abs(self.H_TES_industry) - H_ch
            self.H_TES_industry = -H_ch
            TES_next_industry = 1
        elif TES_next_industry <= self.industry_TES.min_state:
            if state_copy[2,6] * (1 - self.industry_TES.sigma_HS) <= self.industry_TES.min_state:
                H_dis = 0
            else:
                H_dis = (state_copy[2,6] * (1 - self.industry_TES.sigma_HS) * self.industry_TES.max_storage - self.industry_TES.min_storage) * self.industry_TES.yita_dis
            self.H_GB_industry = max(0, abs(self.H_TES_industry) - H_dis)
            self.H_TES_industry = H_dis
            TES_next_industry = self.industry_TES.min_state

        P_EB_residential = self.H_EB_residential / self.residential_EB.yita_EB
        P_EB_commercial = self.H_EB_commercial / self.commercial_EB.yita_EB
        gas_GB_industry = self.H_GB_industry / (self.industry_GB.LHV_gas * self.industry_GB.yita_GB)

        # 计算电交易量：正值表示电力盈余，可作为卖方；负值表示缺电，需要买电。
        self.trade_e_residential = (self.P_HCHP_residential + self.P_CHP_residential + self.P_ESS_residential + self.PV_residential) - \
                              (self.electric_load_residential + P_CER_residential + P_EB_residential)
        self.trade_e_commercial = (self.P_HCHP_commercial + self.P_CHP_commercial + self.P_ESS_commercial + self.PV_commercial) - \
                              (self.electric_load_commercial + P_CER_commercial + P_EB_commercial)
        self.trade_e_industry = (self.P_HCHP_industry + self.P_CHP_industry + self.P_ESS_industry + self.PV_industry) - \
                              self.electric_load_industry


        #进行CDA交易，统计交易数据
        carbon_intensity_residential = self._electric_carbon_intensity(
            self.P_HCHP_residential,
            self.P_CHP_residential,
            self.P_ESS_residential,
            self.PV_residential,
            gas_CHP_residential,
        )
        carbon_intensity_commercial = self._electric_carbon_intensity(
            self.P_HCHP_commercial,
            self.P_CHP_commercial,
            self.P_ESS_commercial,
            self.PV_commercial,
            gas_CHP_commercial,
        )
        carbon_intensity_industry = self._electric_carbon_intensity(
            self.P_HCHP_industry,
            self.P_CHP_industry,
            self.P_ESS_industry,
            self.PV_industry,
            gas_CHP_industry,
        )
        self.carbon_intensity_residential = carbon_intensity_residential
        self.carbon_intensity_commercial = carbon_intensity_commercial
        self.carbon_intensity_industry = carbon_intensity_industry

        if self.market_mechanism in (
            "carbon_aware",
            "carbon_aware_v2",
            "carbon_dividend_v3",
            "carbon_dividend_equal_v3",
        ):
            carbon_price = self._cda_carbon_price(
                [
                    self.trade_e_residential,
                    self.trade_e_commercial,
                    self.trade_e_industry,
                ],
                [
                    carbon_intensity_residential,
                    carbon_intensity_commercial,
                    carbon_intensity_industry,
                ],
            )
            self.current_cda_carbon_price = carbon_price
            if self.market_mechanism in (
                "carbon_dividend_v3",
                "carbon_dividend_equal_v3",
            ):
                sell_remain_e, buy_remain_e, trade_record_e = CDA(
                    self.trade_e_residential,
                    self.price_e_residential,
                    self.trade_e_commercial,
                    self.price_e_commercial,
                    self.trade_e_industry,
                    self.price_e_industry,
                )
            else:
                market = (
                    volume_preserving_carbon_aware_CDA
                    if self.market_mechanism == "carbon_aware_v2"
                    else carbon_aware_CDA
                )
                sell_remain_e, buy_remain_e, trade_record_e = market(
                    self.trade_e_residential,
                    self.price_e_residential,
                    carbon_intensity_residential,
                    self.trade_e_commercial,
                    self.price_e_commercial,
                    carbon_intensity_commercial,
                    self.trade_e_industry,
                    self.price_e_industry,
                    carbon_intensity_industry,
                    carbon_price=carbon_price,
                )
        else:
            self.current_cda_carbon_price = 0.0
            sell_remain_e, buy_remain_e, trade_record_e = CDA(self.trade_e_residential, self.price_e_residential,
                                                                            self.trade_e_commercial, self.price_e_commercial,
                                                                            self.trade_e_industry, self.price_e_industry)

        #统计交易情况
        self.trade_record_e = trade_record_e
        self.trade_record_e_dict = {1:0, 2:0, 3:0}
        self.sell_to_grid_residential = 0
        self.sell_to_grid_commercial = 0
        self.sell_to_grid_industry = 0
        self.buy_from_grid_residential = 0
        self.buy_from_grid_commercial = 0
        self.buy_from_grid_industry = 0

        #sell_remain: 卖剩下的 buy_remain: 还需买的
        sell_remain_e_dict = {1:0, 2:0, 3:0}
        buy_remain_e_dict = {1:0, 2:0, 3:0}

        if len(sell_remain_e) != 0:
            for record in sell_remain_e:
                sell_remain_e_dict[record[0]] = record[1]
        if len(buy_remain_e) != 0:
            for record in buy_remain_e:
                buy_remain_e_dict[record[0]] = record[1]
        if len(trade_record_e) != 0:
            for record in trade_record_e:
                self.trade_record_e_dict[record[0]] += record[2]
                self.trade_record_e_dict[record[1]] -= record[2]

        # 计算各智能体电交易收益
        trade_profit_e = profit(trade_record_e)

        #计算各智能体与电网的交易量
        for key, electricity in sell_remain_e_dict.items():
            if key == 1:
                self.sell_to_grid_residential = abs(electricity)
            elif key == 2:
                self.sell_to_grid_commercial = abs(electricity)
            elif key == 3:
                self.sell_to_grid_industry = abs(electricity)

        for key, electricity in buy_remain_e_dict.items():
            if key == 1:
                self.buy_from_grid_residential = abs(electricity)
            if key == 2:
                self.buy_from_grid_commercial = abs(electricity)
            if key == 3:
                self.buy_from_grid_industry = abs(electricity)

        self.carbon_levy = {1: 0.0, 2: 0.0, 3: 0.0}
        self.carbon_credit = {1: 0.0, 2: 0.0, 3: 0.0}
        self.avoided_carbon_credit_basis = {1: 0.0, 2: 0.0, 3: 0.0}
        if self.market_mechanism in (
            "carbon_dividend_v3",
            "carbon_dividend_equal_v3",
        ):
            self.carbon_levy, self.carbon_credit, self.avoided_carbon_credit_basis = (
                carbon_dividend_settlement(
                    trade_record_e,
                    {
                        1: carbon_intensity_residential,
                        2: carbon_intensity_commercial,
                        3: carbon_intensity_industry,
                    },
                    {
                        1: self.buy_from_grid_residential,
                        2: self.buy_from_grid_commercial,
                        3: self.buy_from_grid_industry,
                    },
                    self.EF,
                    self.current_cda_carbon_price,
                    dividend_mode=(
                        "equal"
                        if self.market_mechanism == "carbon_dividend_equal_v3"
                        else "targeted"
                    ),
                )
            )
        self.carbon_dividend_fund = float(sum(self.carbon_levy.values()))
        self.carbon_dividend_balance_error = float(
            sum(self.carbon_levy.values()) - sum(self.carbon_credit.values())
        )

        #获取系统下一时刻的状态
        if flag == 0:
            if step < 1439:
                next_state_residential = np.append(state_train_residential[step + 1],np.array([ESS_next_residential, TES_next_residential])).reshape(1,-1)
                next_state_commercial = np.append(state_train_commercial[step + 1],np.array([ESS_next_commercial, TES_next_commercial])).reshape(1, -1)
                next_state_industry = np.append(state_train_industry[step + 1],np.array([ESS_next_industry, TES_next_industry])).reshape(1, -1)
                done = 0
            else:
                next_state_residential = np.append(state_train_residential[0],np.array([ESS_next_residential, TES_next_residential])).reshape(1,-1)
                next_state_commercial = np.append(state_train_commercial[0],np.array([ESS_next_commercial, TES_next_commercial])).reshape(1, -1)
                next_state_industry = np.append(state_train_industry[0],np.array([ESS_next_industry, TES_next_industry])).reshape(1, -1)
                done = 1


        if flag == 1:
            self.n += 1
            next_state_residential = np.append(state_test_residential[step + 1], np.array([ESS_next_residential, TES_next_residential])).reshape(1,-1)
            next_state_commercial = np.append(state_test_commercial[step + 1], np.array([ESS_next_commercial, TES_next_commercial])).reshape(1,-1)
            next_state_industry = np.append(state_test_industry[step + 1], np.array([ESS_next_industry, TES_next_industry])).reshape(1,-1)
            if self.n < 24:
                done = 0
            else:
                done = 1
                self.n = 0

        #计算奖励
        cost_residential = - trade_profit_e[1] - self.sell_to_grid_residential * self.electricity_price / 2 + \
                           self.buy_from_grid_residential * self.electricity_price + gas_CHP_residential * self.gas_price \
                           + H2_residential * self.H2_price + self.waste_h_residential \
                           + self.carbon_levy[1] - self.carbon_credit[1]
        carbon_residential = gas_CHP_residential * self.NCV_gas * self.CC * self.OF * self.ratio_carbon_oxygen + self.buy_from_grid_residential * self.EF

        cost_commercial = - trade_profit_e[2] - self.sell_to_grid_commercial * self.electricity_price / 2 + \
                           self.buy_from_grid_commercial * self.electricity_price + gas_CHP_commercial * self.gas_price \
                           + H2_commercial * self.H2_price + self.waste_h_commercial \
                           + self.carbon_levy[2] - self.carbon_credit[2]

        carbon_commercial = gas_CHP_commercial * self.NCV_gas * self.CC * self.OF * self.ratio_carbon_oxygen + self.buy_from_grid_commercial * self.EF

        cost_industry = - trade_profit_e[3] - self.sell_to_grid_industry * self.electricity_price / 2 + \
                           self.buy_from_grid_industry * self.electricity_price + gas_CHP_industry * self.gas_price \
                           + H2_industry * self.H2_price + self.waste_h_industry \
                           + self.carbon_levy[3] - self.carbon_credit[3]
        carbon_industry = (gas_CHP_industry + gas_GB_industry) * self.NCV_gas * self.CC * self.OF * self.ratio_carbon_oxygen + self.buy_from_grid_industry * self.EF

        return next_state_residential, next_state_commercial, next_state_industry, done, \
               np.array([cost_residential, carbon_residential]), np.array([cost_commercial, carbon_commercial]), \
               np.array([cost_industry, carbon_industry])

    def schedule_result_residential(self):
        return (self.electric_load_residential, self.heat_load_residential, self.cooling_load_residential, self.PV_residential, self.P_CHP_residential,
                self.P_HCHP_residential, self.C_AC_residential, self.C_CER_residential, self.H_EB_residential, self.P_ESS_residential,
                self.H_TES_residential, self.buy_from_grid_residential, self.sell_to_grid_residential, self.trade_record_e_dict[1],
                self.price_e_residential, self.waste_h_residential)

    def schedule_result_commercial(self):
        return (self.electric_load_commercial, self.heat_load_commercial, self.cooling_load_commercial, self.PV_commercial, self.P_CHP_commercial,
                self.P_HCHP_commercial, self.C_AC_commercial, self.C_CER_commercial, self.H_EB_commercial, self.P_ESS_commercial,
                self.H_TES_commercial, self.buy_from_grid_commercial,self.sell_to_grid_commercial, self.trade_record_e_dict[2],
                self.price_e_commercial, self.waste_h_commercial)

    def schedule_result_industry(self):
        return (self.electric_load_industry, self.heat_load_industry, self.PV_industry, self.P_CHP_industry, self.P_HCHP_industry,
                self.H_GB_industry, self.C_AC_industry, self.P_ESS_industry, self.H_TES_industry, self.buy_from_grid_industry,
                self.sell_to_grid_industry, self.trade_record_e_dict[3], self.price_e_industry, self.waste_h_industry)

    def reset_train(self):
        a = state_train_residential[0]
        b = state_train_commercial[0]
        c = state_train_industry[0]
        initial_state_residential = np.append(a,np.array([self.residential_ESS.reset(),self.residential_TES.reset()])).reshape(1,-1)
        initial_state_commercial = np.append(b,np.array([self.commercial_ESS.reset(),self.commercial_TES.reset()])).reshape(1,-1)
        initial_state_industry = np.append(c,np.array([self.industry_ESS.reset(),self.industry_TES.reset()])).reshape(1,-1)
        return initial_state_residential, initial_state_commercial, initial_state_industry

    def reset_test(self):
        #测试日期分别为一月二十一，四月二十一，七月二十一，十月二十一
        i = 0 #winter
        # i = 24 #spring
        # i = 48  #summer
        # i = 72  #autumn
        a = state_test_residential[i]
        b = state_test_commercial[i]
        c = state_test_industry[i]
        initial_state_residential = np.append(a,np.array([self.residential_ESS.reset(),self.residential_TES.reset()])).reshape(1,-1)
        initial_state_commercial = np.append(b,np.array([self.commercial_ESS.reset(),self.commercial_TES.reset()])).reshape(1,-1)
        initial_state_industry = np.append(c,np.array([self.industry_ESS.reset(),self.industry_TES.reset()])).reshape(1,-1)

        return initial_state_residential, initial_state_commercial, initial_state_industry, i
