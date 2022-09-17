# fitness_func.py
# =========================================================================================================================
# This module contains a hedging reservoir simulation model and objective function.
#
# =========================================================================================================================
import math
import pandas as pd
import numpy as np

def Storage(param_space, Total_Copacity, lower_Capacity, Initial_Stor, Inflow_TS, Target_12TS, Ration_Ratio_48TS):

    length_timeseries = len(Inflow_TS)
    Sim_storage = np.zeros(length_timeseries)
    Sim_Supply = np.zeros(length_timeseries)
    Sim_Spill_R = np.zeros(length_timeseries)
    Sim_Drought_phase = np.zeros(length_timeseries)
    Sim_Shortage = np.zeros(length_timeseries)
    Sim_Target = np.zeros(length_timeseries)

    for timeseries in list(range(length_timeseries)):
        operation_month = timeseries%12
        con_month = operation_month
        cau_month = operation_month + 12
        alr_month = operation_month + 24
        sev_month = operation_month + 36
        Stand_Supply = Target_12TS[operation_month]
        Con_Supply = Target_12TS[operation_month] * Ration_Ratio_48TS[con_month]
        Cau_Supply = Target_12TS[operation_month] * Ration_Ratio_48TS[cau_month]
        Alr_Supply = Target_12TS[operation_month] * Ration_Ratio_48TS[alr_month]
        Sev_Supply = Target_12TS[operation_month] * Ration_Ratio_48TS[sev_month]

        if timeseries == 0:
            current_storage = Initial_Stor
            Sim_storage[0] = Initial_Stor
        else:
            current_storage = Sim_storage[timeseries]

        Available_water = current_storage + Inflow_TS[timeseries]

        if Available_water-Target_12TS[operation_month]*Ration_Ratio_48TS[sev_month] <= lower_Capacity:
            Sim_Drought_phase[timeseries] = 5
        elif current_storage + Inflow_TS[timeseries] <= param_space[sev_month]:
            Sim_Drought_phase[timeseries] = 4
            Sim_Supply[timeseries] = Sev_Supply
        elif current_storage + Inflow_TS[timeseries] <= param_space[alr_month]:
            Sim_Drought_phase[timeseries] = 3
            Sim_Supply[timeseries] = Alr_Supply
        elif current_storage + Inflow_TS[timeseries] <= param_space[cau_month]:
            Sim_Drought_phase[timeseries] = 2
            Sim_Supply[timeseries] = Cau_Supply
        elif current_storage + Inflow_TS[timeseries] <= param_space[con_month]:
            Sim_Drought_phase[timeseries] = 1
            Sim_Supply[timeseries] = Con_Supply
        else:
            Sim_Supply[timeseries] = Stand_Supply

        if Available_water >= Total_Copacity[operation_month] + Sim_Supply[timeseries]:
            Sim_Spill_R[timeseries] = Available_water - Sim_Supply[timeseries] - Total_Copacity[operation_month]
        else:
            pass


        if timeseries == length_timeseries - 1:
            pass
        else:
            Sim_storage[timeseries+1] = Sim_storage[timeseries] + Inflow_TS[timeseries] - Sim_Supply[timeseries] - Sim_Spill_R[timeseries]

        Sim_Target[timeseries] = Target_12TS[operation_month]
        Sim_Shortage[timeseries] = Target_12TS[operation_month] - Sim_Supply[timeseries]

    Total_Shortage = Sim_Shortage.sum()
    fail_count = Sim_Drought_phase == 5
    fail_penalty = fail_count.sum()
    reverse_penalty = 0
    for oper_month in list(range(12)):
        if param_space[oper_month] <= param_space[oper_month+12]+Target_12TS[oper_month]*Ration_Ratio_48TS[oper_month]*0.5:
            reverse_penalty = reverse_penalty + 1
        elif param_space[oper_month+12] <= param_space[oper_month+24]+Target_12TS[oper_month]*Ration_Ratio_48TS[oper_month+12]*0.5:
            reverse_penalty = reverse_penalty + 1
        elif param_space[oper_month+24] <= param_space[oper_month+36]+Target_12TS[oper_month]*Ration_Ratio_48TS[oper_month+24]*0.5:
            reverse_penalty = reverse_penalty + 1
        elif param_space[oper_month+36] <= lower_Capacity + Target_12TS[oper_month]*Ration_Ratio_48TS[oper_month+36]*0.5:
            reverse_penalty = reverse_penalty + 1
        else:
            pass

    fitness = Total_Shortage + 100000*fail_penalty + 100000000*reverse_penalty
    return fitness
