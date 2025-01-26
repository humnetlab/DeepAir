import os

testGroup_list = [8, 5, 9, 0, 1, 2, 3, 4, 6, 7]

"""
Evaluate performance across different groups and attempt to remove certain features.
"""
testGroup_list = [8, 5, 9, 0, 1, 2, 3, 4, 6, 7]
data_dir = "./pretrainMergedData"
merged_data_dir = "preTrain_shapeNoDrop_allGroups"
kind = "preProcess"
for testGroup in testGroup_list:
    run_cmd = f"python model_GBM_pretrainMergedData_allGroups.py --data_dir {data_dir} --merged_data_dir {merged_data_dir} --testGroup {testGroup} --kind {kind} --appendix delFea"
    print(f"\nRunning cmd is {run_cmd}\n")
    os.system(run_cmd)
