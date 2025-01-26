"""
Utilize CNN to deal with Elevation data and LandCover data.
Encode Elevation and LandCover information from the raw data using the CNN structure from the trained model.
Generate new data for subsequent use in the GBM model.
When selecting different model structures, the imported model needs to be modified.
"""

import argparse
import gc
import glob
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

print("pytorch version : ", torch.__version__)

import sys

sys.path.append("../")
import data_loader_group
import load_static_data
import utils
from logger import Logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")  # 'cpu' in this case

print("Using device : ", DEVICE)

dataPath = "./data/DeepAir/"
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--testGroup", type=int)
parser.add_argument("--weight_save_dir", type=str, default="saved_weights_years")
parser.add_argument("--model_save_dirs", type=str, default=None)
parser.add_argument("--case_label", type=str, default=None)
parser.add_argument("--loss_func", type=str, default=None)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--outData_save_dirs", type=str, default=None)

# log file name
parser.add_argument("--log_file", type=str, default="PM25")
args = parser.parse_args()
config = json.load(open("config.json", "r"))


# select importedmodel
if args.model_save_dirs == "ElevaLand.preTrain.shapeNoDrop":
    from situ_DNN_preTrain_shapeNoDrop import *


def get_model(args, kwargs):
    if args.case_label == "ElevaLand":
        model = situNet_ElevaLand(args, **kwargs)
    return model


def get_calseKwargs(caseLabel):
    if args.case_label == "ElevaLand":
        kwargs = get_kwargs(situNet_ElevaLand)
    return kwargs


# we keep uncertainty and the final prediction for each sample
# for each testing file, we load 20 models and predict the testing samples
def dataTransfer(args, elogger, testGroup=0):
    print(" --------- ")
    print("Running CNN embedding on Group %d in case %s ..." % (testGroup, args.suffix))
    trainTestData = []

    # load train data
    print(f"Reading datas")
    for part in range(6):
        filePath = (
            dataPath
            + "STData/train_G"
            + str(testGroup)
            + "/train_"
            + str(part).zfill(2)
        )
        if os.path.exists(filePath):
            partData = data_loader_group.MySet(args, filePath)
            trainTestData += partData.content
            del partData
        gc.collect()

    # load the testing data
    # 1 testing file
    filePath = dataPath + "STData/test_G" + str(testGroup) + "/test_00"
    partData = data_loader_group.MySet(args, filePath)
    trainTestData += partData.content
    del partData
    gc.collect()
    print("# data number : %d" % len(trainTestData))

    # load all basemap images
    baseMapDict = {}
    print("%d Basemap Images loaded!" % len(baseMapDict))

    # load all elevation data
    elevationDict = load_static_data.loadElevation()
    print(f"{len(elevationDict)} Elevation data loaded!")

    # load all emission data
    emissionDict = {}
    print("Emission loaded!")

    # load all landcover images
    landCoverImagesDict = load_static_data.loadLandCoverImgs()
    # landCoverImagesDict = {}
    print("%d LandCover Images loaded!" % len(landCoverImagesDict))

    # Load model
    directory = f"./{args.model_save_dirs}/{args.weight_save_dir}"
    allFiles = sorted(
        glob.glob(directory + "/PM25_G" + str(testGroup) + "_" + args.suffix + "*"),
        reverse=True,
    )
    if args.model_save_dirs.split(".")[-1] == "timeSplit":
        allFiles = sorted(
            glob.glob(directory + "/PM25_" + args.suffix + "*"), reverse=True
        )
    print("# of models for ensemble : %d" % len(allFiles))

    data_iter = data_loader_group.get_loader_fromData(
        trainTestData, args.batch_size, shufData=False
    )

    # one model predict all
    # collect results of the varRun*50 running
    monitor_lons = []
    monitor_lats = []

    model_parameter_file = allFiles[0]

    kwargs = get_calseKwargs(args.case_label)
    model = get_model(args, kwargs)
    if torch.cuda.is_available():
        model.cuda()

    modelFile = model_parameter_file
    print(f"weight file is {modelFile}")
    model.load_state_dict(torch.load(modelFile, map_location=DEVICE))
    print("Model loaded!")

    model.eval()
    transferedData = torch.Tensor()
    for idx, attr in enumerate(data_iter):
        attr["Elevation"] = load_static_data.addElevationToBatchData(
            attr, elevationDict
        )
        attr["LandCoverImg"] = load_static_data.addLandCoverToBatchData(
            attr, landCoverImagesDict
        )
        attr = utils.to_var(attr)
        numSamplesInBatch = attr["PM2.5"].data.cpu().size()[0]
        if numSamplesInBatch == 1:
            continue

        lons = np.asarray(attr["Lon"].data.cpu()).flatten()
        lons = utils.unnormalize(lons, "Lon")
        lons = np.asarray(lons)
        lons = np.round(lons, 2)
        lons = list(lons)
        lats = np.asarray(attr["Lat"].data.cpu()).flatten()
        lats = utils.unnormalize(lats, "Lat")
        lats = np.round(lats, 2)
        lats = list(lats)
        monitor_lons.extend(lons)
        monitor_lats.extend(lats)
        # Encode Landcover and Elevation based on pretrained model
        temp_transferData = model.data_transfer(attr, config)
        if idx == 0:
            transferedData = temp_transferData
        else:
            transferedData = torch.cat([transferedData, temp_transferData], dim=0)

    transferedData = transferedData.detach().cpu().numpy()
    print(f"transferedData shape is {transferedData.shape}")
    outColumns = ["yearID", "dayIdx", "weekday", "month", "Lon", "Lat", "basin_ID"]

    # define Embedding shape and Generate columns
    if args.model_save_dirs == "ElevaLand.preTrain.shapeNoDrop":
        Elevation_embed_dim = 3
        LandCover_embed_dim = 3

    Elevation_cols = [f"Elevation_{i}" for i in range(Elevation_embed_dim)]
    landCover_cols = [f"LandCover_{i}" for i in range(LandCover_embed_dim)]
    outColumns.extend(Elevation_cols)
    outColumns.extend(landCover_cols)

    outFrame = pd.DataFrame(columns=outColumns, data=transferedData)

    for col in ["yearID", "dayIdx", "weekday", "month", "basin_ID"]:
        outFrame[col] = outFrame[col].astype("int")
    outFrame["yearID"] = outFrame["yearID"] + 2006
    outFrame["yearID"] = outFrame["yearID"].astype("str")
    outFrame["dayIdx"] = outFrame["dayIdx"].astype("str")
    outFrame["dayIdx"] = outFrame["dayIdx"].apply(lambda x: x.zfill(3))
    outFrame["Lon"] = outFrame["Lon"].apply(lambda x: utils.unnormalize(x, "Lon"))
    outFrame["Lat"] = outFrame["Lat"].apply(lambda x: utils.unnormalize(x, "Lat"))
    for col in ["Lon", "Lat"]:
        outFrame[col] = outFrame[col].apply(lambda x: "%.2f" % x)

    embed_csv_savePath = (
        f"{args.csv_savePath}/{args.case_label}_allGroup_testG{args.testGroup}.csv"
    )

    print(f"outFrame data dtypes is: {outFrame.dtypes}")
    outFrame.to_csv(embed_csv_savePath, index=0)
    print(f"\nSuccessfully save CSV file to {embed_csv_savePath}")
    print(f"CSV shape is {outFrame.shape}")


def get_kwargs(model_class):
    model_args = list(inspect.signature(model_class.__init__).parameters)
    shell_args = args._get_kwargs()
    kwargs = dict(shell_args)
    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs


def run():
    # get the model arguments
    kwargs = get_calseKwargs(args.case_label)

    print(f"args are {args}")
    print(f"kwargs are {kwargs}")

    testGroup = args.testGroup

    print(
        "Testing group : %s; case : %s; DataTransfer process ; suffix : %s"
        % (testGroup, args.case_label, args.suffix)
    )

    args.csv_savePath = (
        f"./pretrain_allGroup_embededData/{args.outData_save_dirs}/G{testGroup}"
    )
    if not os.path.exists(args.csv_savePath):
        os.makedirs(args.csv_savePath)

    args.log_file = "PM25_dataTransfer"
    # save cmd input
    elogger = Logger(f"{args.csv_savePath}/PM25_embedData_G{testGroup}.log")
    cmd_input = "python " + " ".join(sys.argv) + "\n"
    elogger.log(cmd_input)
    dataTransfer(args, elogger, testGroup=testGroup)


if __name__ == "__main__":
    run()

"""
python main_pretrainedCNN2data.py --testGroup 5 --model_save_dirs ElevaLand.ES.decayLR --weight_save_dir saved_earlyStop_weights_yearsG5 --case_label ElevaLand --loss_func Huber
"""
