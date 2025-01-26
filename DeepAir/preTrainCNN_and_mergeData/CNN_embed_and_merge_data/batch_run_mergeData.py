"""
File    :   batch_run_mergeData.py
Desc    :   merge CNN transfered data and situData
"""

import os

testGroups = [5, 9, 0, 1, 2, 3, 4, 6, 7, 8]

""" Generate embed data corresponding to different pre-trained models by group (10 pre-trained models * 10 corresponding datasets). """
model_save_dirs = "ElevaLand.preTrain.shapeNoDrop"
outData_saveDir = "preTrain_shapeNoDrop_allGroups"

for test_group in testGroups:
    # pretrain CNN
    getCNNEmbed_cmd = f"python main_pretrainedCNN2data_allGroups.py --testGroup {test_group} --model_save_dirs {model_save_dirs} --weight_save_dir saved_earlyStop_weights_yearsG{test_group} --case_label ElevaLand --loss_func Huber --outData_save_dirs {outData_saveDir}"
    print("\nRunning cmd is:\n{}\n\n".format(getCNNEmbed_cmd))
    os.system(getCNNEmbed_cmd)

    # concate data
    mergeData_cmd = f"python main_dataTransMerge_allGroups.py --embedData_save_dirs {outData_saveDir} --testGroup {test_group} --mergedData_save_dir {outData_saveDir} --raw_situData situData_years"
    print("\nRunning cmd is:\n{}\n\n".format(mergeData_cmd))
    os.system(mergeData_cmd)
