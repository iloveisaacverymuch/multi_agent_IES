class CHP():
    def __init__(self, CHP_MAX):
        #热电联产相关参数
        self.yita_CHP_E = 0.45 #热电联产的电效率
        self.yita_CHP_H = 0.4 #热电联产的热效率
        self.CHP_max = CHP_MAX  #热电联产最大出力电功率
        self.LHV_gas = 10.0 #天然气热值10kwh/m3

    def CHP(self, P_CHP):
        #输入是电功率比率，输出是电功率,热功率和氢气体积
        #热功率
        H_CHP = P_CHP * self.CHP_max * self.yita_CHP_H / self.yita_CHP_E
        gas = H_CHP / self.LHV_gas / self.yita_CHP_H
        return P_CHP * self.CHP_max, H_CHP, gas

class HCHP():
    def __init__(self, HCHP_MAX):
        self.HCHP_max = HCHP_MAX #氢热电联产的最大电功率
        self.yita_HCHP_E = 0.45 #热电联产的电效率
        self.yita_HCHP_H = 0.4 #热电联产的热效率
        self.LHV_H2 = 3.54  # 氢能热值

    def HCHP(self, P_HCHP):
        H_HCHP = P_HCHP * self.HCHP_max * self.yita_HCHP_H / self.yita_HCHP_E
        H2 = H_HCHP / self.LHV_H2 / self.yita_HCHP_H
        return P_HCHP * self.HCHP_max, H_HCHP, H2

class GB():
    def __init__(self, GB_MAX):
        #燃气锅炉相关参数
        self.yita_GB = 0.8  #燃气锅炉的效率
        self.GB_max = GB_MAX  #燃气锅炉最大热出力
        self.LHV_gas = 10.0 #天然气热值10kwh/m3
    def GB(self, H_GB):
        # 输入是热功率比率，输出是热功率和燃气体积
        G_GB = H_GB * self.GB_max / (self.LHV_gas * self.yita_GB)
        return H_GB * self.GB_max, G_GB


class EB():
    def __init__(self, EB_MAX):
        self.yita_EB = 0.95 #电锅炉的效率
        self.EB_max = EB_MAX  #电锅炉的最大热效率

    def EB(self, H_EB):
        #输入是热功率比率，输出是电功率和热功率
        P_EB = H_EB * self.EB_max / self.yita_EB
        return H_EB * self.EB_max, P_EB


class AC():
    def __init__(self, AC_MAX):
        self.COP_AC = 0.7 #吸收式制冷机效率
        self.AC_max = AC_MAX #吸收式制冷机最大冷功率

    def AC(self, C_AC):
        H_AC = C_AC * self.AC_max / self.COP_AC
        return C_AC * self.AC_max, H_AC

class CER():
    def __init__(self, CER_MAX):
        self.COP_CER = 4.0 #压缩式制冷机效率
        self.CER_max = CER_MAX #压缩式制冷机最大冷功率

    def CER(self, CER):
        P_CER = CER * self.CER_max / self.COP_CER
        return CER * self.CER_max, P_CER

class ESS():
    def __init__(self, max_storage, min_storage, charge_max):
        #电储能相关参数
        self.yita_charge = 0.95 #充电效率
        self.yita_discharge = 0.95  #放电效率
        self.max_storage = max_storage #最大储电量
        self.min_storage = min_storage #最小储电量
        self.charge_max = charge_max  #最大充放功率
        self.delta_t = 1    #时间间隔，单位h
        self.min_state = 0.1

    #电储能出力 P_ESS表示充电或放电功率
    def P_ESS(self, ESS_power):
        return ESS_power * self.charge_max

    #电储能模型状态转移 ESS_power表示充电或放电功率(取绝对值)，SOC为荷电状态
    def state_transmission(self, ch_ess, ESS_power, SOC):
        if ch_ess >= 0:
            SOC_next = (SOC * self.max_storage - ESS_power * self.delta_t / self.yita_discharge) / self.max_storage
        else:
            SOC_next = (SOC * self.max_storage + ESS_power * self.delta_t * self.yita_charge) / self.max_storage

        return SOC_next

    #重置电储能初始状态
    def reset(self):
        return self.min_state

#热存储
class TES():
    def __init__(self, max_storage, charge_max):
        self.yita_ch = 0.9  #储热效率
        self.yita_dis = 0.9 #放热效率
        self.sigma_HS = 0.1 #热量损耗率
        self.charge_max = charge_max  #最大储放热功率
        self.max_storage = max_storage #最大储热量
        self.min_storage = 0  #最小储热量
        self.delta_t = 1    #时间间隔，单位h
        self.min_state = self.min_storage / self.max_storage

    # 热储能出力 TES_power表示储热或放热功率
    def H_TES(self, TES_power):
        return TES_power * self.charge_max

    # 热储能模型状态转移 THS_power表示储热或放热功率(取绝对值)，TES为热储能状态
    def state_transmission(self, ch_tes, TES_power, TES):
        #如果当前状态为放热
        if ch_tes >= 0:
            THS_next = (TES * (1 - self.sigma_HS) * self.max_storage - TES_power * self.delta_t / self.yita_dis) / self.max_storage
        #如果当前状态为储热
        else:
            THS_next = (TES * (1 - self.sigma_HS) * self.max_storage + TES_power * self.delta_t * self.yita_ch) / self.max_storage

        return THS_next

    #重置储热状态
    def reset(self):
        return self.min_state