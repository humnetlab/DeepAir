"""
Merge the transformed CSV data with the original CSV data and perform preliminary processing.

"""

import torch
import os
import pandas as pd
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if use_cuda else 'cpu')   # 'cpu' in this case
print("Using device : ", DEVICE)

dataPath = './data/DeepAir/'

parser = argparse.ArgumentParser()

parser.add_argument('--testGroup', type=int)
parser.add_argument('--embedData_save_dirs', type = str, default = None)
# log file name
parser.add_argument('--suffix', type=str, default="")
parser.add_argument('--mergedData_save_dir', type=str, default=None)
parser.add_argument('--raw_situData', type=str, default=None)

args = parser.parse_args()


def run():
    testGroup = args.testGroup
    transferDataPath = f"./pretrain_allGroup_embededData/{args.embedData_save_dirs}/G{args.testGroup}/ElevaLand_allGroup_testG{testGroup}.csv"

    transfered_df = pd.read_csv(transferDataPath)
    transfered_df["yearID"] = transfered_df["yearID"].astype('str')
    transfered_df["dayIdx"] = transfered_df["dayIdx"].astype('str').apply(lambda x:x.zfill(3) )
    transfered_df["yearDay"] = transfered_df["yearID"] + transfered_df["dayIdx"]
    transfered_df["Lon_Lat"] = transfered_df["Lon"].apply(lambda x:"%.2f" %x).astype(str)+ \
                                "_" +transfered_df["Lat"].apply(lambda x:"%.2f" %x).astype(str)
    transfered_df["mergeIndex"] = transfered_df["yearDay"] + "_" + transfered_df["Lon_Lat"]

    transfered_df_dropCols = ["yearID", "dayIdx", "weekday", "month", "Lon", "Lat", "basin_ID", "yearDay", "Lon_Lat"]
    transfered_df.drop(transfered_df_dropCols, axis=1, inplace=True)

    situData_path = dataPath + f"STData/{args.raw_situData}.csv"
    situData_df = pd.read_csv(situData_path)
    situData_df['yearDay'] = situData_df['yearDay'].astype('str')
    situData_df["Lon_Lat"] = situData_df["Lon"].apply(lambda x:"%.2f" %x).astype(str)+ \
                                "_" +situData_df["Lat"].apply(lambda x:"%.2f" %x).astype(str)
    # generate merge index based on date_location 
    situData_df["mergeIndex"] = situData_df["yearDay"] + "_" + situData_df["Lon_Lat"]
    situ_drop_cols = [f"LandCover_{str(i).zfill(2)}" for i in range(20)]
    situ_drop_cols.extend(["Elevation", "Lon_Lat"])
    situData_df.drop(situ_drop_cols, axis=1, inplace=True)

    newData = pd.merge(situData_df, transfered_df, how='left', on = 'mergeIndex')
    newData.drop(["mergeIndex"], axis=1, inplace=True)
    newData['yearDay'] = newData['yearDay'].astype('int')
    print(newData.dtypes)
    print(newData.shape)

    if not os.path.exists(f"./mergedData/{args.mergedData_save_dir}"):
        os.makedirs(f"./mergedData/{args.mergedData_save_dir}")
    newData.to_csv(f"./mergedData/{args.mergedData_save_dir}/merged_situData_allGroup_byTestG{args.testGroup}_years.csv", index=0)

if __name__ == '__main__':
    run()
