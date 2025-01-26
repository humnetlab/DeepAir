"""use CNN to deal with Elevation data and LandCover data"""

import argparse
import gc
import glob
import inspect
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import r2_score

print("pytorch version : ", torch.__version__)
import sys

import torch.optim as optim

sys.path.append("../")
import data_loader_group
import load_static_data
import utils
from EarlyStopping import EarlyStopping
from logger import Logger
from situ_DNN_shapeNoDrop import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")  # 'cpu' in this case

print("Using device : ", DEVICE)


dataPath = "./data/DeepAir/"

parser = argparse.ArgumentParser()
# basic args
parser.add_argument("--task", type=str)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=250)
parser.add_argument("--clip", type=int, default=1)
parser.add_argument("--testGroup", type=int)
parser.add_argument("--weight_save_dir", type=str, default="saved_weights_years")
parser.add_argument("--data_years", type=str, default="logs_years")
parser.add_argument("--test_years", type=str, default="years")
parser.add_argument("--save_dirs", type=str, default=None)
parser.add_argument("--numSave", type=int, default=50)

parser.add_argument("--case_label", type=str, default=None)  # "met3DEEESA"
parser.add_argument("--loss_func", type=str, default=None)  # "MSE"
parser.add_argument("--suffix", type=str, default="")  # "met3DEEESA"

# evaluation args
parser.add_argument("--weight_file", type=str)
parser.add_argument("--result_file", type=str)


# log file name
parser.add_argument("--log_file", type=str, default="PM25")

args = parser.parse_args()

config = json.load(open("config.json", "r"))


testGroup = 0


def get_model(args, kwargs):
    if args.case_label == "ElevaLand":
        model = situNet_ElevaLand(args, **kwargs)
    return model


def get_calseKwargs(caseLabel):
    if args.case_label == "ElevaLand":
        kwargs = get_kwargs(situNet_ElevaLand)
    return kwargs


def write_result(fs, label, pred, pred_var):
    for i in range(len(pred)):
        fs.write("%.3f,%.3f,%.3f\n" % (label[i], pred[i], pred_var[i]))


def mae(label, pred):
    label = np.array(label)
    pred = np.array(pred)
    err = np.abs(pred - label)
    return np.mean(err)


def mape(label, pred):
    label = np.array(label)
    pred = np.array(pred)
    nonZeroIdx = label > 0
    label = label[nonZeroIdx]
    pred = pred[nonZeroIdx]
    err = np.divide(np.abs(pred - label), label)
    return np.mean(err)


def mape_large(label, pred):
    label = np.array(label)
    pred = np.array(pred)
    nonZeroIdx = label > 10.0
    label = label[nonZeroIdx]
    pred = pred[nonZeroIdx]
    err = np.divide(np.abs(pred - label), label)
    return np.mean(err)


def rmse(label, pred):
    label = np.array(label)
    pred = np.array(pred)
    loss = np.square(pred - label)
    loss = np.mean(loss)
    return np.sqrt(loss)


def getMonitorR2(df, ordered_monitors, monitor_col="monitors"):
    R2_dict = {}
    for monitor in ordered_monitors:
        temp_df = df[df[monitor_col] == monitor]
        R2_dict[monitor] = r2_score(temp_df["labels"], temp_df["preds"])
    return R2_dict


# load the training data and testing data in memory to save time
def train_fast(model, elogger, testGroup, earlyStop_elogger):
    print(f"#### Training: testGroup is {testGroup} ####")
    train_data = []
    # 6 training file
    print("Loading train data......")
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
            train_data += partData.content
            del partData
        gc.collect()

    print("Loading test data......")
    # 1 testing file
    filePath = dataPath + "STData/test_G" + str(testGroup) + "/test_00"
    partData = data_loader_group.MySet(args, filePath)
    eval_data = partData.content
    del partData
    gc.collect()

    print("# samples for training : %d" % len(train_data))
    print("# samples for testing : %d" % len(eval_data))

    # load all basemap images
    baseMapDict = {}
    print("%d Basemap Images loaded!" % len(baseMapDict))

    # load all elevation data
    elevationDict = load_static_data.loadElevation()
    print("Elevation loaded!")

    # load all emission data
    emissionDict = {}
    print("Emission loaded!")

    # load all landcover images
    landCoverImagesDict = load_static_data.loadLandCoverImgs()
    print("%d LandCover Images loaded!" % len(landCoverImagesDict))

    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()

    para_count = 0
    weights_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            para_count += 1
            weights_count += param.data.numel()
    print("Total number of parameters : ", para_count, weights_count)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # Define sheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.9
    )  # learning rate decay

    # change of error / R2
    errorChange_training = []
    errorChange_testing = []
    R2Change_training = []
    R2Change_testing = []

    MAPEChange_training = []
    MAPEChange_testing = []
    MAEChange_training = []
    MAEChange_testing = []
    RMSEChange_training = []
    RMSEChange_testing = []
    xIdx_training = []
    xIdx_testing = []

    numR = 1

    early_stop_modelName = (
        f"{args.log_file}_G{args.testGroup}_{args.suffix}_earlyStopping"
    )
    early_stop_modelSavePath = (
        f"./{args.save_dirs}/{args.weight_save_dir}/{early_stop_modelName}"
    )
    early_stopping = EarlyStopping(
        patience=50,
        verbose=True,
        delta=0,
        path=early_stop_modelSavePath,
        trace_func=earlyStop_elogger.log,
    )

    for epoch in range(args.epochs):

        print("\n========================")
        print("Training on epoch {} / {}".format(epoch, args.epochs))

        model.train()

        # split data to batches
        data_iter = data_loader_group.get_loader_fromData(train_data, args.batch_size)

        running_loss = 0.0
        training_labels = []
        training_preds = []
        count = 0
        for idx, attr in enumerate(data_iter):
            # add basemap images to attr
            attr["Elevation"] = load_static_data.addElevationToBatchData(
                attr, elevationDict
            )
            attr["LandCoverImg"] = load_static_data.addLandCoverToBatchData(
                attr, landCoverImagesDict
            )

            # transform the input to pytorch variable
            attr = utils.to_var(attr)
            numSamplesInBatch = attr["PM2.5"].data.cpu().size()[0]
            if numSamplesInBatch == 1:
                continue

            count += 1

            pred_dict, loss = model.eval_on_batch(attr, config)

            pred = pred_dict["pred"].data.cpu().numpy().flatten()
            label = pred_dict["label"].data.cpu().numpy().flatten()
            training_labels.extend(label.tolist())
            training_preds.extend(pred.tolist())

            r2 = r2_score(training_labels, training_preds)
            err_mape = mape(training_labels, training_preds)
            err_mae = mae(training_labels, training_preds)
            err_rmse = rmse(training_labels, training_preds)

            # update the model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            running_loss += loss.item()
            print(
                "\r Progress {:.4f}%, average loss = {:.4f}, R2 = {:.4f}, MAPE = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}".format(
                    (idx + 1) * 100.0 / len(data_iter),
                    running_loss / (idx + 1.0),
                    r2,
                    err_mape,
                    err_mae,
                    err_rmse,
                )
            )

        idx = count - 1

        scheduler.step()

        print()
        elogger.log(
            "Training Epoch {}, Loss {:.4f}, R2 = {:.4f}, MAPE = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}".format(
                epoch, running_loss / (idx + 1.0), r2, err_mape, err_mae, err_rmse
            )
        )

        errorChange_training.append(running_loss / (idx + 1.0))
        R2Change_training.append(r2)
        MAPEChange_training.append(err_mape)
        MAEChange_training.append(err_mae)
        RMSEChange_training.append(err_rmse)
        xIdx_training.append(epoch)

        # plot the change of error
        fig = plt.figure(figsize=(4, 3))
        plt.plot(
            xIdx_training, errorChange_training, lw=1, color="#b10026", label="train"
        )
        if len(errorChange_testing) > 0:
            plt.plot(
                xIdx_testing, errorChange_testing, lw=1, color="#005a32", label="test"
            )
        plt.title("Training Huber error")
        plt.ylabel("Huber error")
        plt.xlabel("Epoch")
        plt.ylim(0)
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            "./{}/{}/error_G{}_training_{}.png".format(
                args.save_dirs, args.data_years, testGroup, args.suffix
            ),
            dpi=150,
        )
        plt.close()

        # plot the change of R2
        fig = plt.figure(figsize=(4, 3))
        plt.plot(xIdx_training, R2Change_training, lw=1, color="#252525", label="train")
        if len(R2Change_testing) > 0:
            plt.plot(
                xIdx_testing, R2Change_testing, lw=1, color="#005a32", label="test"
            )
        plt.title("Training R2 score")
        plt.ylabel("R2")
        plt.xlabel("Epoch")
        plt.ylim(-0.1, 1)
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            "./{}/{}/R2_G{}_training_{}.png.png".format(
                args.save_dirs, args.data_years, testGroup, args.suffix
            ),
            dpi=150,
        )
        plt.close()

        # plot the change of MAPE
        fig = plt.figure(figsize=(4, 3))
        plt.plot(
            xIdx_training, MAPEChange_training, lw=1, color="#252525", label="train"
        )
        if len(MAPEChange_testing) > 0:
            plt.plot(
                xIdx_testing, MAPEChange_testing, lw=1, color="#005a32", label="test"
            )
        plt.title("Training MAPE")
        plt.ylabel("MAPE")
        plt.xlabel("Epoch")
        plt.ylim(0)
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            "./{}/{}/MAPE_G{}_training_{}.png".format(
                args.save_dirs, args.data_years, testGroup, args.suffix
            ),
            dpi=150,
        )
        plt.close()

        # plot the change of MAE
        fig = plt.figure(figsize=(4, 3))
        plt.plot(
            xIdx_training, MAEChange_training, lw=1, color="#252525", label="train"
        )
        if len(MAEChange_testing) > 0:
            plt.plot(
                xIdx_testing, MAEChange_testing, lw=1, color="#005a32", label="test"
            )
        plt.title("Training MAE")
        plt.ylabel("MAE")
        plt.xlabel("Epoch")
        plt.ylim(0)
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            "./{}/{}/MAE_G{}_training_{}.png".format(
                args.save_dirs, args.data_years, testGroup, args.suffix
            ),
            dpi=150,
        )
        plt.close()

        # plot the change of RMSE
        fig = plt.figure(figsize=(4, 3))
        plt.plot(
            xIdx_training, RMSEChange_training, lw=1, color="#252525", label="train"
        )
        if len(RMSEChange_testing) > 0:
            plt.plot(
                xIdx_testing, RMSEChange_testing, lw=1, color="#005a32", label="test"
            )
        plt.title("Training RMSE")
        plt.ylabel("RMSE")
        plt.xlabel("Epoch")
        plt.ylim(0)
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            "./{}/{}/RMSE_G{}_training_{}.png".format(
                args.save_dirs, args.data_years, testGroup, args.suffix
            ),
            dpi=150,
        )
        plt.close()

        # evaluate
        if epoch % numR == 0:
            # evaluate the model after each epoch
            print(" ======== Evaluation =========")
            res_eval = evaluate_fast(
                model,
                baseMapDict,
                elevationDict,
                emissionDict,
                landCoverImagesDict,
                elogger,
                eval_data,
                epoch=epoch,
                save_result=False,
            )
            loss_eval, r2_test, mape_test, mae_test, rmse_test = res_eval

            errorChange_testing.append(loss_eval)
            R2Change_testing.append(r2_test)
            MAPEChange_testing.append(mape_test)
            MAEChange_testing.append(mae_test)
            RMSEChange_testing.append(rmse_test)
            xIdx_testing.append(epoch)

            early_stopping(epoch, loss_eval, model)

            if early_stopping.early_stop:
                print(f"Early stopping in Epoch {epoch}")
                break
    print("============== train finished! =================")


def evaluate_fast(
    model,
    baseMapDict,
    elevationDict,
    emissionDict,
    landCoverImagesDict,
    elogger,
    eval_data,
    epoch=0,
    save_result=False,
):
    print(" --------- ")
    print("Running test on Group %d ..." % (testGroup))
    model.train()
    if save_result:
        fs = open("%s" % args.result_file, "w")

    labels = []
    preds = []

    varRun = 10

    running_loss = 0.0
    data_iter = data_loader_group.get_loader_fromData(eval_data, args.batch_size)

    count = 0

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

        count += 1

        loss_ = []
        for v in range(varRun):
            pred_dict, loss = model.eval_on_batch(attr, config)
            loss_.append(loss.data.item())
            if v == 0:
                label = pred_dict["label"].data.cpu().numpy().flatten()
                pred = pred_dict["pred"].data.cpu().numpy().flatten()
            else:
                pred_ = pred_dict["pred"].data.cpu().numpy().flatten()
                pred = np.vstack([pred, pred_])

        if varRun > 1:
            pred_var = np.std(pred, axis=0)
            pred = np.mean(pred, axis=0)
        else:
            pred_var = 0

        labels.extend(label.tolist())
        preds.extend(pred.tolist())

        if save_result:
            write_result(fs, label, pred, pred_var)

        running_loss += np.mean(loss_)

    idx = count - 1

    # r2
    print("number of samples : %d -- %d" % (len(labels), len(preds)))
    loss_eval = running_loss / (idx + 1.0)
    r2 = r2_score(labels, preds)
    err_mape = mape(labels, preds)
    err_mae = mae(labels, preds)
    err_rmse = rmse(labels, preds)
    print(
        "Evaluate performance, loss {:.4f}, R2 = {:.4f}, MAPE = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}".format(
            running_loss / (idx + 1.0), r2, err_mape, err_mae, err_rmse
        )
    )
    elogger.log(
        "Evaluate performance, Loss {:.4f}, R2 = {:.4f}, MAPE = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}".format(
            running_loss / (idx + 1.0), r2, err_mape, err_mae, err_rmse
        )
    )

    if save_result:
        fs.close()

    return loss_eval, r2, err_mape, err_mae, err_rmse


# we keep uncertainty and the final prediction for each sample
# for each testing file, we load 20 models and predict the testing samples
def ensembleTest_fast(args, elogger, testGroup=0):
    print(" --------- ")
    print("Running test on Group %d in case %s ..." % (testGroup, args.suffix))

    # load the testing data
    # 1 testing file
    filePath = dataPath + "STData/test_G" + str(testGroup) + "/test_00"
    partData = data_loader_group.MySet(args, filePath)
    eval_data = partData.content
    del partData
    gc.collect()

    print("# samples for testing : %d" % len(eval_data))

    # load all basemap images
    baseMapDict = {}
    print("%d Basemap Images loaded!" % len(baseMapDict))

    # load all elevation data
    elevationDict = load_static_data.loadElevation()
    print("Elevation loaded!")

    # load all emission data
    emissionDict = {}
    print("Emission loaded!")

    # load all landcover images
    landCoverImagesDict = load_static_data.loadLandCoverImgs()
    # landCoverImagesDict = {}
    print("%d LandCover Images loaded!" % len(landCoverImagesDict))

    # mode list
    directory = f"./{args.save_dirs}/{args.weight_save_dir}"
    allFiles = sorted(
        glob.glob(directory + "/PM25_G" + str(testGroup) + "_" + args.suffix + "*"),
        reverse=True,
    )

    print("# of models for ensemble : %d" % len(allFiles))

    monitor_lons = []
    monitor_lats = []
    labels = []
    preds = []
    preds_var = []

    varRun = 100

    data_iter = data_loader_group.get_loader_fromData(
        eval_data, args.batch_size, shufData=False
    )

    count = 0

    # one model predict all
    # collect results of the varRun*50 running
    yearDay_list = []
    monitor_lons = []
    monitor_lats = []
    labels = []
    preds = []

    for m in range(len(allFiles)):
        if m > args.numSave:
            break
        print("model ---- ", m)
        kwargs = get_calseKwargs(args.case_label)
        model = get_model(args, kwargs)
        if torch.cuda.is_available():
            model.cuda()

        modelFile = allFiles[m]
        print(f"weight file is {modelFile}")
        model.load_state_dict(torch.load(modelFile, map_location=DEVICE))

        # repeat the model for varRun times
        for v in range(varRun):
            print(" run  ---- ", v)
            model.train()
            # predict all data
            pred_this = []
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

                count += 1

                # ===== run multiple models for ensemble prediction on this batch of data ======
                # save the average prediction, average uncertainty
                # the first run
                if m == 0 and v == 0:
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
                    temp_yearDay_list = np.asarray(attr["yearDay"].data.cpu()).flatten()
                    yearDay_list.extend(temp_yearDay_list)

                pred_dict, loss = model.eval_on_batch(attr, config)
                if m == 0 and v == 0:
                    label = pred_dict["label"].data.cpu().numpy().flatten()
                    labels.extend(label.tolist())

                pred = pred_dict["pred"].data.cpu().numpy().flatten()
                pred_this.extend(pred.tolist())

            # all prediction in this running has been collected
            preds.append(pred_this)

    # dimention of preds should be varRun * 50
    preds_var = np.std(np.asarray(preds), axis=0)
    preds_avg = np.mean(np.asarray(preds), axis=0)
    preds_avg = preds_avg.tolist()

    # r2
    print("number of samples : %d -- %d" % (len(labels), len(preds_avg)))
    r2 = r2_score(labels, preds_avg)
    err_mape = mape(labels, preds_avg)
    err_mape_large = mape_large(labels, preds_avg)
    err_mae = mae(labels, preds_avg)
    err_rmse = rmse(labels, preds_avg)
    print(
        "Evaluate performance, R2 = {:.4f}, MAPE = {:.4f}, MAPE_large = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}".format(
            r2, err_mape, err_mape_large, err_mae, err_rmse
        )
    )
    elogger.log(
        "Evaluate performance, R2 = {:.4f}, MAPE = {:.4f}, MAPE_large = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}".format(
            r2, err_mape, err_mape_large, err_mae, err_rmse
        )
    )

    monitor_locs = []
    for i in range(len(monitor_lons)):
        loc = str(monitor_lons[i]) + "_" + str(monitor_lats[i])
        monitor_locs.append(loc)

    res_pd = pd.DataFrame(
        {"labels": labels, "preds": preds_avg, "monitors": monitor_locs}
    )

    # scatter plot
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    plt.scatter(labels, preds_avg, s=2, alpha=0.2, cmap=plt.get_cmap("jet"))
    ax.text(20, 80, "R2: $%.3f$" % r2, size=14, ha="center", va="center")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Prediction")
    plt.tight_layout()
    plt.savefig(
        "./{}/{}/prediction_G{}_test_{}.png".format(
            args.save_dirs, args.data_years, testGroup, args.suffix
        ),
        dpi=150,
    )
    plt.close()

    # scatter plot
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    ordered_monitors = sorted(set(res_pd["monitors"]))
    g = sns.FacetGrid(res_pd, col="monitors", col_order=ordered_monitors, col_wrap=4)
    kws = dict(s=20, linewidth=0.5, edgecolor=None, alpha=0.3)
    map_g = (
        g.map(plt.scatter, "labels", "preds", color="g", **kws)
        .set(xlim=(0, 60), ylim=(0, 60))
        .fig.subplots_adjust(wspace=0.15, hspace=0.15)
    )

    R2_dict = getMonitorR2(res_pd, ordered_monitors, monitor_col="monitors")
    for index, ax in g.axes_dict.items():
        ax.text(
            15, 40, "R2: $%.3f$" % R2_dict[index], size=12, ha="center", va="center"
        )

    fig.set_figheight(6)
    fig.set_figwidth(6)
    plt.savefig(
        "./{}/{}/prediction_G{}_test_grid_{}.png".format(
            args.save_dirs, args.data_years, testGroup, args.suffix
        ),
        dpi=150,
    )
    plt.close()

    if not os.path.exists(f"./{args.save_dirs}/predResults"):
        os.makedirs(f"./{args.save_dirs}/predResults")
    savePath = f"./{args.save_dirs}/predResults/ensembleTest_G{testGroup}.csv"
    monitor_lons = np.round(monitor_lons, 2)
    monitor_lats = np.round(monitor_lats, 2)
    labels = np.round(labels, 3)
    preds_avg = np.round(preds_avg, 3)
    preds_var = np.round(preds_var, 4)
    save_df = pd.DataFrame(
        {
            "yearDay": yearDay_list,
            "testGroup": testGroup,
            "monitor_lon": monitor_lons,
            "monitor_lat": monitor_lats,
            "label": labels,
            "pred_avg": preds_avg,
            "preds_var": preds_var,
        }
    )
    save_df.to_csv(savePath, index=0)

    certainTest_fast(
        args,
        allFiles,
        data_iter,
        elevationDict,
        emissionDict,
        landCoverImagesDict,
        testGroup,
    )

    print("\n\nEnsemble Test and Certainly Test are finished!")
    print("Bingo!")


def certainTest_fast(
    args,
    allFiles,
    data_iter,
    elevationDict,
    emissionDict,
    landCoverImagesDict,
    testGroup,
):

    elogger = Logger(args.save_dirs, args.data_years, "PM25_test_certain", testGroup)

    monitor_lons = []
    monitor_lats = []
    yearDay_list = []
    labels = []
    preds = []

    monitor_lons = []
    monitor_lats = []
    labels = []
    preds = []

    kwargs = get_calseKwargs(args.case_label)
    model = get_model(args, kwargs)
    if torch.cuda.is_available():
        model.cuda()
    modelFile = allFiles[0]
    print(f"weight file is {modelFile}")
    model.load_state_dict(torch.load(modelFile, map_location=DEVICE))

    print("Begin certain prediction process")
    model.eval()
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
        temp_yearDay_list = np.asarray(attr["yearDay"].data.cpu()).flatten()
        yearDay_list.extend(temp_yearDay_list)

        pred_dict, loss = model.eval_on_batch(attr, config)
        label = pred_dict["label"].data.cpu().numpy().flatten()
        labels.extend(label.tolist())

        pred = pred_dict["pred"].data.cpu().numpy().flatten()
        preds.extend(pred.tolist())

    # r2
    print("number of samples : %d -- %d" % (len(labels), len(preds)))
    r2 = r2_score(labels, preds)
    err_mape = mape(labels, preds)
    err_mape_large = mape_large(labels, preds)
    err_mae = mae(labels, preds)
    err_rmse = rmse(labels, preds)
    print(
        "Evaluate performance, R2 = {:.4f}, MAPE = {:.4f}, MAPE_large = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}".format(
            r2, err_mape, err_mape_large, err_mae, err_rmse
        )
    )
    elogger.log(
        "Evaluate performance, R2 = {:.4f}, MAPE = {:.4f}, MAPE_large = {:.4f}, MAE = {:.4f}, RMSE = {:.4f}".format(
            r2, err_mape, err_mape_large, err_mae, err_rmse
        )
    )

    monitor_locs = []
    for i in range(len(monitor_lons)):
        loc = str(monitor_lons[i]) + "_" + str(monitor_lats[i])
        monitor_locs.append(loc)

    res_pd = pd.DataFrame({"labels": labels, "preds": preds, "monitors": monitor_locs})
    # scatter plot
    fig = plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)
    plt.scatter(labels, preds, s=2, alpha=0.2, cmap=plt.get_cmap("jet"))
    ax.text(20, 80, "R2: $%.3f$" % r2, size=16, ha="center", va="center")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Prediction")
    plt.tight_layout()
    plt.savefig(
        "./{}/{}/prediction_G{}_test_certain_{}.png".format(
            args.save_dirs, args.data_years, testGroup, args.suffix
        ),
        dpi=150,
    )
    plt.close()

    # scatter plot
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    ordered_monitors = sorted(set(res_pd["monitors"]))
    g = sns.FacetGrid(res_pd, col="monitors", col_order=ordered_monitors, col_wrap=4)
    kws = dict(s=20, linewidth=0.5, edgecolor=None, alpha=0.3)
    map_g = (
        g.map(plt.scatter, "labels", "preds", color="g", **kws)
        .set(xlim=(0, 60), ylim=(0, 60))
        .fig.subplots_adjust(wspace=0.15, hspace=0.15)
    )
    R2_dict = getMonitorR2(res_pd, ordered_monitors, monitor_col="monitors")
    for index, ax in g.axes_dict.items():
        ax.text(
            15, 40, "R2: $%.3f$" % R2_dict[index], size=12, ha="center", va="center"
        )
    fig.set_figheight(6)
    fig.set_figwidth(6)
    plt.savefig(
        "./{}/{}/prediction_G{}_test_certain_grid_{}.png".format(
            args.save_dirs, args.data_years, testGroup, args.suffix
        ),
        dpi=150,
    )
    plt.close()

    if not os.path.exists(f"./{args.save_dirs}/predResults"):
        os.makedirs(f"./{args.save_dirs}/predResults")
    savePath = f"./{args.save_dirs}/predResults/certainTest_G{testGroup}.csv"
    monitor_lons = np.round(monitor_lons, 2)
    monitor_lats = np.round(monitor_lats, 2)
    labels = np.round(labels, 3)
    preds = np.round(preds, 3)
    save_df = pd.DataFrame(
        {
            "yearDay": yearDay_list,
            "testGroup": testGroup,
            "monitor_lon": monitor_lons,
            "monitor_lat": monitor_lats,
            "label": labels,
            "pred_PM": preds,
        }
    )
    save_df.to_csv(savePath, index=0)


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
    caseLabel = args.case_label

    print(
        "Testing group : %s; case : %s; task : %s ; suffix : %s"
        % (testGroup, args.case_label, args.task, args.suffix)
    )

    # Create dirs
    if not os.path.exists("./{}/{}".format(args.save_dirs, args.weight_save_dir)):
        os.makedirs("./{}/{}".format(args.save_dirs, args.weight_save_dir))
    if not os.path.exists("./{}/{}".format(args.save_dirs, args.data_years)):
        os.makedirs("./{}/{}".format(args.save_dirs, args.data_years))

    # model instance
    model = get_model(args, kwargs)

    if torch.cuda.is_available():
        model.cuda()

    if args.task == "train":
        # experiment logger
        args.log_file = "PM25"
        elogger = Logger(args.save_dirs, args.data_years, args.log_file, testGroup)
        cmd_input = "python " + " ".join(sys.argv) + "\n"
        elogger.log(cmd_input)

        earlyStop_elogger = Logger(
            args.save_dirs,
            args.data_years,
            args.log_file,
            "{}_earlyStopping".format(testGroup),
        )
        train_fast(
            model, elogger, testGroup=testGroup, earlyStop_elogger=earlyStop_elogger
        )

    elif args.task == "ensembleTest":
        args.log_file = "PM25_test"
        elogger = Logger(args.save_dirs, args.data_years, args.log_file, testGroup)
        cmd_input = "python " + " ".join(sys.argv) + "\n"
        elogger.log(cmd_input)
        ensembleTest_fast(args, elogger, testGroup=testGroup)


if __name__ == "__main__":
    run()
