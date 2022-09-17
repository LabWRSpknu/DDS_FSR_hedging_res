# -*- coding: utf-8 -*-
import DDS_FSR
import toolkit as util
import os, glob, shutil
import numpy as np
import pandas as pd
import math as m
import time
#======================== Brief Description ====================================
# This is the main calling script for the dynamically dimensioned search allowing
# flexible search range (DDS-FSR) to derive hedging rule curves.
# The DDS-FSR is modified the Dynamically  dimensioned search (DDS) by Jin Y.
# The DDS-FSR is purposed to efficiently solve constrained optimization problems.
# The DDS-FSR can recursively update search ranges of decision variables with limited overlaps.
# This Dynamically Dimensioned Search (DDS) algorithm by Bryan Tolson,
# Department of Civil & Environmental Engineering University of Waterloo
# The DDS was originally coded on Nov 2005 by Bryan Tolson
# This Python-distribution DDS-FSR is modified DDS-Py.
# The DDS-Py was coded on Aug 2015 by Thouheed Abdul Gaffoor.

# REFERENCE FOR THIS ALGORITHM:
# Tolson, B. A., and C. A. Shoemaker (2007), Dynamically dimensioned search algorithm
# for computationally efficient watershed model calibration, Water Resour. Res., 43,
# W01413, doi:10.1029/2005WR004723.
#===============================================================================
####################################################################################
Inflow_Name_list = ['ADIH', 'HC', 'NG']
# Inflow_Name_list = ['ADIH']
drop_index = np.arange(0,12)
project_path = os.getcwd()
project_list = os.listdir(project_path)
Hist_list = []   # reservoir historical data file index in project folder
Input_list = []   # reservoir simulation input data file index in project folder
for i_Jin in list(range(len(project_list))):
    if project_list[i_Jin].find('00_Hist') != -1:
        j_Jin = i_Jin
        Hist_list.append(j_Jin)
    if project_list[i_Jin].find('Input') != -1:
        k_Jin = i_Jin
        Input_list.append(k_Jin)
##### read initial storage
Tot_Init_Stor = []   # Initial storage [ADIH, HC, NG]
for i_Jin, j_Jin in enumerate(Hist_list):
    Hist_file_path = project_path + '\\' + project_list[j_Jin]
    Hist_read = pd.read_csv(Hist_file_path)
    Hist_read = Hist_read.drop(index=drop_index).reset_index()
    Hist_read['Date'] = pd.to_datetime(Hist_read['Date'])
    # target_index = Hist_read.index[Hist_read['Date'] == Oper_Start_date].tolist()
    Starting_stor = Hist_read['Storage'][[0]].values[0]
    Tot_Init_Stor.append(Starting_stor)
##### read reservoir capacity & low water level storage
Tot_Res_low_stor = []
Tot_Res_Cap = np.zeros((12, len(Inflow_Name_list)))
Tot_Res_target = np.zeros((12, len(Inflow_Name_list)))
Tot_Res_Ration = np.zeros((48, len(Inflow_Name_list)))
Tot_Res_Trig = np.zeros((48, len(Inflow_Name_list)))
for i_Jin, j_Jin in enumerate(Input_list):
    Input_file_path = project_path + '\\' + project_list[j_Jin]
    Input_read = pd.read_csv(Input_file_path)
    raw_cap = Input_read['Capacity'].dropna().values
    raw_target = Input_read['Plan_Supply'].dropna().values
    raw_ration = Input_read['Ration_Ratio'].values
    raw_trig = Input_read['Trig_Vol'].values
    Tot_Res_Cap[:, i_Jin] = raw_cap[:12]
    Tot_Res_low_stor.append(raw_cap[13])
    Tot_Res_target[:, i_Jin] = raw_target
    Tot_Res_Ration[:, i_Jin] = raw_ration
    Tot_Res_Trig[:, i_Jin] = raw_trig
##### read reservoir inflows
path_dds_input_file = project_path + '\\' + 'DDS_inp.txt'
path_dds_initial_file = project_path + '\\' + 'initials.txt'
path_dds_bound_file = project_path + '\\' + 'Storage.txt'

for i_Jin, j_Jin in enumerate(Hist_list):
    Hist_file_path = project_path + '\\' + project_list[j_Jin]
    Hist_read = pd.read_csv(Hist_file_path)
    Hist_read = Hist_read.drop(index=drop_index).reset_index()
    Hist_read['Date'] = pd.to_datetime(Hist_read['Date'])
    num_days = Hist_read['Date'].dt.days_in_month.values
    raw_inflow_data = Hist_read['Inflow_cms'].values
    for k_jin in np.arange(len(raw_inflow_data)):
        raw_inflow_data[k_jin] = raw_inflow_data[k_jin] * 86400 * num_days[k_jin] / 10**6
    Init_Stor = Tot_Init_Stor[i_Jin]
    Res_Cap = Tot_Res_Cap[:, i_Jin]
    Res_Low = Tot_Res_low_stor[i_Jin]
    Res_Target = Tot_Res_target[:, i_Jin]
    Res_Ration = Tot_Res_Ration[:, i_Jin]
    Res_Trig = Tot_Res_Trig[:, i_Jin]

    Scen_inflow = raw_inflow_data
    Scen_Compact_name = Inflow_Name_list[i_Jin]+'_Hist_DDS'
    # Scen_Compact_name = '{:<30}'.format(Scen_Compact_name)
    print(Scen_Compact_name)
    Open_DDS_Input = open(path_dds_input_file, 'r')
    Read_DDS_Input = Open_DDS_Input.readlines()
    no2_line_script = Read_DDS_Input[3].split('#')
    Read_DDS_Input[3] = Scen_Compact_name + '#' + no2_line_script[1]
    Open_DDS_Input.close()
    Open_DDS_Input = open(path_dds_input_file, 'w')
    Open_DDS_Input.writelines(Read_DDS_Input)
    Open_DDS_Input.close()
    Open_DDS_Bound = open(path_dds_bound_file, 'r')
    Read_DDS_Bound = Open_DDS_Bound.readlines()
    Read_DDS_Bound = [Read_DDS_Bound[0]]
    Open_DDS_Bound.close()
    Open_DDS_Bound = open(path_dds_bound_file, 'w')
    for k_Jin in list(range(len(Res_Trig))):
        Bound_Name = '{:<3}'.format('x' + str(k_Jin + 1))
        Bound_Upper_value = '{:>8.2f}'.format(round(Res_Cap[k_Jin % 12], 2))
        Bound_Lower_value = '{:>8.2f}'.format(round(Res_Low, 2))
        Bound_Disc_Flag = '{:>3d}'.format(int(0))
        Bound_total_script = Bound_Name + Bound_Lower_value + Bound_Upper_value + Bound_Disc_Flag + '\n'
        Read_DDS_Bound.append(Bound_total_script)
    Open_DDS_Bound.writelines(Read_DDS_Bound)
    Open_DDS_Bound.close()
    Open_DDS_Initial = open(path_dds_initial_file, 'r')
    Read_DDS_Initial = Open_DDS_Initial.readlines()
    Read_DDS_Initial = Read_DDS_Initial[:2]
    Open_DDS_Initial.close()
    Open_DDS_Initial = open(path_dds_initial_file, 'w')
    for k_Jin in list(range(len(Res_Trig))):
        Initial_value = str('{:<10.3f}'.format(Res_Trig[k_Jin])) + '\n'
        Read_DDS_Initial.append(Initial_value)
    Open_DDS_Initial.writelines(Read_DDS_Initial)
    Open_DDS_Initial.close()

    DDS_inp = util.read_DDS_inp('DDS_inp.txt')  # read 1

    bounds_file = DDS_inp['objfunc_name'] + '.txt'
    DV_bounds = util.read_param_file(bounds_file)  # read 2
    num_dec = DV_bounds['S_min'].shape[0]  # number of dec variables
    # ===============================================================================
    # 2.0   Input verification

    # Ensure valid entry for parallel processing slaves
    # n= 1: serial run, n = 0: optimised auto slaves, n > 1 = 'n' user specified slaves
    assert DDS_inp[
               'num_slaves'] >= 0, 'For a parallel run, please enter a valid number (> 1) of processing slaves! Try program again.'

    # Determine if Parallel or serial execution:
    if DDS_inp['num_slaves'] > 1:
        # parallel run with 'n' user specified slaves
        parallel_run = True
        # total iters = n slaves * evaluations per slave
        DDS_inp['num_iters'] = DDS_inp['num_iters'] * DDS_inp['num_slaves']
        # number of initial solution iters = number of slaves (each slave gets one eval)
        its = DDS_inp['num_slaves']
    elif DDS_inp['num_slaves'] == 0:
        parallel_run = True
        # total iters = n slaves * evaluations per slave
        DDS_inp['num_iters'] = DDS_inp['num_iters'] * DDS_inp['num_slaves']
        # number of initial solution iters = number of slaves (each slave gets one eval)
        its = DDS_inp['num_slaves']
    elif DDS_inp['num_slaves'] == 1:
        parallel_run = False
        # number of initial solution iters = max of 5 and 0.5% of total iterations
        its = max(5, np.int64(np.around(0.005 * DDS_inp['num_iters'])))

    assert DDS_inp['obj_flag'] == -1 or DDS_inp[
        'obj_flag'] == 1, 'Please enter -1 or 1 for objective function flag!  Try program again.'

    assert DDS_inp['num_trials'] >= 1 and DDS_inp[
        'num_trials'] <= 1000, 'Please enter 1 to 1000 optimization trials!  Try program again.'

    assert DDS_inp['num_iters'] >= 7 and DDS_inp[
        'num_iters'] <= 1000000, 'Please enter 7 to 1000000 for max # function evaluations!  Try program again.'

    assert DDS_inp['out_print'] == 0 or DDS_inp[
        'out_print'] == 1, 'Please enter 0 or 1 for output printing flag! Try program again.'

    # Set random seed
    # np.random.seed(DDS_inp['user_seed'])

    # Initial Solution Set-up
    if DDS_inp['ini_name'] != '0':  # Case where initial sols file is provided
        its = 1
        Init_Mat = np.loadtxt(DDS_inp['ini_name'], dtype=float, comments='#', skiprows=2)
    # assert DDS_inp['num_trials'] == Init_Mat.shape[0], 'Number of initial solutions does not match # trials selected. Try program again.'
    # assert num_dec == Init_Mat.shape[1], 'Number of dec vars in S_min & initial solution matrix not consistent.'

    # ===============================================================================
    # 3.0   Definition of directory and model subdirectory structure

    # Set script directory - location of this script
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)

    # Get objective function file type by splitting ext. (i.e. 'myfunc'.'exe')
    ext_name = DDS_inp['objfunc_name'].split('.')

    if len(ext_name) == 1:
        # Python objective function (*.py)
        exe_name = np.array([])
    # sets executable file name variable to null
    else:  # case where .exe or .bat file is called
        exe_name = DDS_inp['objfunc_name']
        # indicates that toolkit function ext_function will run to handle .exe file
        DDS_inp['objfunc_name'] = 'ext_function'

    # If subdirectory for model is not specified:
    if DDS_inp['modeldir'] == '0':
        Modeldir = script_dir
    # If relative path specified and parallel run
    elif DDS_inp['modeldir'] != 0 and parallel_run is True:
        # Set Model subdirectory
        Modeldir = os.path.join(script_dir, DDS_inp['modeldir'])
        # Generate copies of base-model for slave-acess
        util.generate_dir(DDS_inp['num_slaves'], Modeldir)
    # Else relative path and serial run
    else:
        Modeldir = os.path.join(script_dir, DDS_inp['modeldir'])
    # ===============================================================================
    # 4.0   Define Output files and arrays:

    filenam1 = DDS_inp['runname'] + '_ini.out'  # initial sol'n file name
    filenam2 = DDS_inp['runname'] + '_AVG.out'  # avg trial outputs
    filenam3 = DDS_inp['runname'] + '_sbest.out'  # output best DV solutions per trial
    filenam4 = DDS_inp['runname'] + '_trials.out'  # output Jbest per iteration number per trial

    # Master output solution Matrix - Initial iteration = initial solution eval
    master_output = np.empty((DDS_inp['num_iters'] + its, 3 + num_dec), dtype=float)
    initial_sols = np.empty((its, 3 + num_dec), dtype=float)
    output = np.empty((DDS_inp['num_iters'], 3), dtype=float)
    # tracks only Jbest but for all trials in one file
    Jbest_trials = np.empty((DDS_inp['num_iters'], DDS_inp['num_trials']), dtype=float)
    # Matrix holding the best sets of decision variables
    Sbest_trials = np.empty((DDS_inp['num_trials'], num_dec), dtype=float)
    sum_output = np.empty((DDS_inp['num_iters'] - its, 3), dtype=float)
    # Matrix holding averages
    MAT_avg = np.empty_like(sum_output)
    # ===============================================================================
    # 5.0   Main Algorithm Calling Loop
    # trial 여러번 할때 활성화 하기 !!=====================================================================================
    # global special_Init
    # special_Init = Init_Mat[:].copy()
    # trial 여러번 할때 활성화 하기 !!=====================================================================================
    for j in range(0, DDS_inp['num_trials']):
        # print(special_Init)
        # Output to console:
        print('Trial number %s executing ... ' % (j + 1))

        # Start timer:
        t_0 = time.time()

        # Feed initial solutions:
        print(DDS_inp['ini_name'])
        # print(Init_Mat[:])
        if DDS_inp['ini_name'] == '0':
            sinitial = np.array([])
        else:
            # sinitial = Init_Mat[j,:]
            sinitial = Init_Mat[:]
            # sinitial = special_Init[:] # trial 여러번 할 경우에 활성화===========================
        # print(sinitial)
        # Call either Serial or MPI DDS Algorithm:
        # print(Init_Mat[:])
        if parallel_run == False:
            output = DDS_FSR.DDS_serial(DDS_inp['objfunc_name'], exe_name, Modeldir, DDS_inp['obj_flag'], DV_bounds,
                                    sinitial,
                                    its, DDS_inp['num_iters'],
                                    Res_Cap, Res_Low, Init_Stor, Scen_inflow, Res_Target, Res_Ration)
        # else:
        # output = DDS.DDS_MPI()

        # store initial solution results
        initial_sols = output['Master'][0:its, :]

        # store truncated outputs - Columns: 0 -> iter #; 1 -> Jbest; 2 -> Jtest
        trunc_out = output['Master'][its:, 0:3]
        output_ALL = output['Master'][its:, :]

        # accumlate only Jbest for each trial:
        Jbest_trials[:, j] = output['Master'][:, 1]
        # accumulate only Sbest for each trial:
        Sbest_trials[j, :] = output['Best_sol']

        if DDS_inp['out_print'] == 0:
            # Write Master Output Matrix at every trial (i.e. - 'Ex1_trial_1.out'):
            # ---------------------------------------------------------------------
            master_file = DDS_inp['runname'] + '_trial_' + str(j + 1) + '.out'
            np.savetxt(master_file, output['Master'])

            # Write Dec. Var. best solutions at every trial (i.e. - 'sbest_trial_1.out'):
            # --------------------------------------------------------------------------
            sbest_file = 'sbest' + '_trial_' + str(j + 1) + '.out'
            np.savetxt(sbest_file, output['Best_sol'])

            # Write Initial Solution to 'Ex1_ini_1.out':
            # ----------------------------------------
            ini_file = DDS_inp['runname'] + '_ini_' + str(j + 1) + '.out'
            np.savetxt(ini_file, initial_sols)

            # Write Jbest compressed output file (i.e. - 'Jbest_trial_1.out'):
            # ----------------------------------------------------------------
            Jbest_file = 'Jbest_trial_' + str(j + 1) + '.out'
            np.savetxt(Jbest_file, Jbest_trials)

        # Prepare matrix for average performance evaluation:
        sum_output = sum_output + trunc_out

        # Stop trial timer
        t_1 = time.time()
        runtime = t_1 - t_0

        # Output to console
        print('Best objective function value of %f found at Iteration %i \n' % (output['F_Best'], output['Best_iter']))
        print('Time of execution for Trial %i was %f seconds or %f hours. \n\n' % (j + 1, runtime, runtime / 3600))
    # ============================================================================
    # 6.0   Post Processing

    # Generate average results from all trials
    Jbest_avg = np.divide(sum_output[:, 1], DDS_inp['num_trials'])
    Jtest_avg = np.divide(sum_output[:, 2], DDS_inp['num_trials'])
    MAT_avg[:, 0] = range(DDS_inp['num_iters'] - its)
    MAT_avg[:, 1] = Jbest_avg
    MAT_avg[:, 2] = Jtest_avg

    # Write averages to output file
    avg_file = DDS_inp['runname'] + '_trial_avgs' + '.out'
    np.savetxt(avg_file, MAT_avg)

    # Generate output directory
    outpath = os.path.join(script_dir, (DDS_inp['runname'] + '_Output'))
    if os.path.exists(outpath):  # If output directory exists - empty it
        exis_files = glob.glob(os.path.join(outpath, "*.out"))
        for f in exis_files:
            os.remove(f)
    else:
        os.makedirs(outpath)  # Else nonexistent - make new output directory

    # Move output files to a new directory
    out_files = glob.iglob(
        os.path.join(script_dir, "*.out"))  # return an iterator of files with ext of *.out in script directory
    for file in out_files:
        if os.path.isfile(file):
            shutil.move(file, outpath)  # move to output file directory
# ============================================================================