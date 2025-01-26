"""
File    :   batch_run.py
Desc    :   Run main_elevaLand_ES_shapeNoDrop.py for every testGroup 
            and get corresponding pretrained models
"""

import os

testGroups = [8, 5, 9, 0, 1, 2, 3, 4, 6, 7]
epoch = 250

for test_group in testGroups:
    test_cmd = f"python main_elevaLand_ES_shapeNoDrop.py --task ensembleTest --epochs {epoch} --testGroup {test_group} --save_dirs ElevaLand.preTrain.shapeNoDrop --weight_save_dir saved_earlyStop_weights_yearsG{test_group} --data_years mergeGroupTest --case_label ElevaLand --loss_func Huber --suffix 3E3L "
    print("\nRunning cmd is:\n{}\n\n".format(test_cmd))
    os.system(test_cmd)
