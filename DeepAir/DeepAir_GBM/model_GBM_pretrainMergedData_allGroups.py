"""
File    :   model_GBM_pretrainMergedData.py
Desc    :   For 'timeSplit' type data, the model used during data generation was based on a temporal split 
        (2017 as the test set, and the remaining years as the training set).
"""

"""Train and test GBM model on Situ data"""
import argparse
import os
import random

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from situDataPreprocess import *
from sklearn.metrics import r2_score
from utils import *

CA_boundary = [-124.48, -114.13, 32.53, 42.01]
CA_boundary_10km = [-124.5, -114.1, 32.5, 42.1]
gridWidth = 0.01

metFeatures = [
    "TEMP",
    "Humidity",
    "PRESS",
    "uWIND",
    "vWIND",
    "Evaporation",
    "Precipitation",
]

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./pretrainMergedData")
parser.add_argument(
    "--merged_data_dir", type=str, help="dirs that save different kind data"
)
parser.add_argument("--testGroup", type=str)
parser.add_argument(
    "--kind", type=str, default="preProcess", help="Use data prerocess or not"
)
parser.add_argument("--y_trans", type=str, default="raw", help="log trans to y or not")
parser.add_argument("--outlier_detection", default="False", action="store_true")
parser.add_argument(
    "--appendix", default=None, type=str, help="To distinguish different version Code"
)

args = parser.parse_args()

args.dataPath = f"{args.data_dir}/{args.merged_data_dir}/merged_situData_allGroup_byTestG{args.testGroup}_years.csv"
args.saveDir = f"./GBM/{args.merged_data_dir}_{args.kind}/{args.testGroup}"
if args.appendix:
    args.saveDir = (
        f"./GBM/{args.merged_data_dir}_{args.kind}_{args.appendix}/{args.testGroup}"
    )
args.res_CSV_savePath = (
    f"{args.saveDir}/situ_GBM_G{args.testGroup}_{args.merged_data_dir}.csv"
)


def log_trans(data):
    ret = np.log(data + 1)
    return ret


def unLog_trans(data):
    ret = np.exp(data) - 1
    return ret


def eval_on_batch_mape(pred, label):
    label_index = label > 0
    loss = np.abs(pred[label_index] - label[label_index]) / label[label_index]
    return np.mean(loss)


def eval_on_batch_mae(pred, label):
    loss = np.abs(pred - label)
    return np.mean(loss)


def eval_on_batch_rmse(pred, label):
    loss = np.square(pred - label)
    loss = np.mean(loss)
    return np.sqrt(loss)


from pyod.models.copod import COPOD


def removeOutlier(df, contamination=0.1):
    clf = COPOD(contamination=contamination)
    clf.fit(df)
    y_scores = clf.decision_scores_  # raw outlier scores
    y_label = clf.labels_
    normal_index = np.argwhere(y_label == 0).reshape(-1)
    normal_df = df.loc[normal_index, :].reset_index(drop=True)
    return normal_df


def trainGBM(df_train, df_test, testGroup, bar=10.0):
    print(df_train.shape)
    print(df_test.shape)

    test_monitors = df_test

    X_drop_cols = ["PM2.5", "yearDay", "Lon", "Lat"]

    y_train = df_train["PM2.5"]
    y_test = df_test["PM2.5"]
    X_train = df_train.drop(X_drop_cols, axis=1)
    X_test = df_test.drop(X_drop_cols, axis=1)

    # change data type
    y_train = y_train.astype("float")
    y_test = y_test.astype("float")

    y_train_large_index = y_train >= bar
    y_test_large_index = y_test >= bar

    # add categorical Feature
    if args.kind == "raw":
        categorical_feature = ["month", "weekDay", "basinID"]
    elif args.kind == "preProcess":
        # categorical_feature = ["month", "weekday", "basinID", "season", "raw_WIND_rank", "Fire_label"]
        if args.appendix == "delFea":
            categorical_feature = ["month", "weekDay"]
        else:
            categorical_feature = [
                "month",
                "weekDay",
                "season",
                "raw_WIND_rank",
                "Fire_label",
            ]

    # construct different dataset
    if args.y_trans == "raw":
        lgb_train = lgb.Dataset(
            X_train, y_train, categorical_feature=categorical_feature
        )
        lgb_eval = lgb.Dataset(
            X_test, y_test, categorical_feature=categorical_feature, reference=lgb_train
        )
    elif args.y_trans == "log":
        lgb_train = lgb.Dataset(
            X_train, log_trans(y_train), categorical_feature=categorical_feature
        )
        lgb_eval = lgb.Dataset(
            X_test,
            log_trans(y_test),
            categorical_feature=categorical_feature,
            reference=lgb_train,
        )

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": ["l1", "huber"],
        "num_iterations": 100000,
        "num_leaves": 512,
        "max_bin": 512,
        "max_depth": 256,
        "learning_rate": 0.01,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 20,
        "early_stopping_rounds": 100,
        "verbose": 0,
        "num_threads": 8,
        "seed": 123,
    }

    print("Starting training...")
    evals_result_dict = {}
    # train
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        evals_result=evals_result_dict,
        verbose_eval=False,
    )
    # plot loss and feature importance
    plotFigs(lgb, gbm, evals_result_dict, testGroup)

    print("Saving model...")
    # save model
    if not os.path.exists(f"{args.saveDir}/model"):
        os.makedirs(f"{args.saveDir}/model")
    gbm.save_model(f"{args.saveDir}/model/GBM_model_G{testGroup}.txt")

    print(X_train.columns)

    print("Starting predicting on training data...")
    # predict
    y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    y_pred = y_pred.astype("float")
    if args.y_trans == "log":
        y_pred = unLog_trans(y_pred)
    # eval
    train_MAE = "%.4f" % eval_on_batch_mae(y_pred, y_train)
    train_RMSE = "%.4f" % eval_on_batch_rmse(y_pred, y_train)
    train_R2 = "%.4f" % r2_score(y_train, y_pred)
    train_MAPE = "%.4f" % eval_on_batch_mape(y_pred, y_train)
    train_large_MAPE = "%.4f" % eval_on_batch_mape(
        y_pred[y_train_large_index], y_train[y_train_large_index]
    )
    print("Train: The MAE of prediction is:", train_MAE)
    print("Train: The RMSE of prediction is:", train_RMSE)
    print("Train: The R2 of prediction is:", train_R2)
    print("Train: The MAPE of prediction is:", train_MAPE)
    print("Train: The Large MAPE of prediction is:", train_large_MAPE)

    print("Starting predicting...")
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred = y_pred.astype("float")

    if args.y_trans == "log":
        y_pred = unLog_trans(y_pred)
    # eval
    test_MAE = "%.4f" % eval_on_batch_mae(y_pred, y_test)
    test_RMSE = "%.4f" % eval_on_batch_rmse(y_pred, y_test)
    test_R2 = "%.4f" % r2_score(y_test, y_pred)
    test_MAPE = "%.4f" % eval_on_batch_mape(y_pred, y_test)
    test_large_MAPE = "%.4f" % eval_on_batch_mape(
        y_pred[y_test_large_index], y_test[y_test_large_index]
    )
    print("Test: The MAE of prediction is:", test_MAE)
    print("Test: The rmse of prediction is:", test_RMSE)
    print("Test: The r2 of prediction is:", test_R2)
    print("Test: The mape of prediction is:", test_MAPE)
    print("Test: The Large Mape of prediction is:", test_large_MAPE)

    # draw scatter plot to show R2
    test_monitors.loc[:, "pred_PM2.5"] = y_pred
    test_monitors.loc[:, "monitors"] = (
        test_monitors["Lon"].apply(lambda x: "%.2f" % x).astype("str")
        + "_"
        + test_monitors["Lat"].apply(lambda x: "%.2f" % x).astype("str")
    )
    test_monitors.loc[:, "recordID"] = (
        test_monitors["yearDay"].astype("str") + "_" + test_monitors["monitors"]
    )
    plotR2Scatters(test_monitors, testGroup=testGroup)

    # save predictions
    if not os.path.exists(f"{args.saveDir}/pred_results/"):
        os.makedirs(f"{args.saveDir}/pred_results/")
    test_monitors.to_csv(
        f"{args.saveDir}/pred_results/prediction_G{testGroup}.csv", index=0
    )

    result_list = [
        train_R2,
        train_MAE,
        train_RMSE,
        train_MAPE,
        train_large_MAPE,
        test_R2,
        test_MAE,
        test_RMSE,
        test_MAPE,
        test_large_MAPE,
    ]
    return result_list, y_test, y_pred


# train GBM from different years' data
def trainGBM_CV_groupYears(bar=10.0):
    allData = []
    print(f"Reading data from {args.dataPath}")
    situData = open(args.dataPath, "r")
    header = situData.readline().rstrip()
    header = header.split(",")
    for row in situData:
        row = row.rstrip().split(",")
        row = [float(r) for r in row]
        allData.append(row)
    random.shuffle(allData)
    allData = pd.DataFrame(allData, columns=header)
    print(f"\noriginal data shape is {allData.shape}")
    allData = dropOutlierRecords(allData)  # drop outlier observations
    # data preprocess
    if args.kind == "preProcess":
        allData = featureTransform(allData, args)
        allData = dropColumns(allData, args)

    print(f"\nAfter preprocess, allData columns are:\n", allData.columns)

    print(f"preProessed data shape is {allData.shape}")

    with open(f"{args.saveDir}/testG{args.testGroup}_dataLog.txt", "w") as f:
        f.write(
            f"After preprocess, allData columns are:\n{allData.columns}",
        )
        f.write("\n\n")
        f.write(f"preProessed data shape is {allData.shape}")
        f.flush()

    GBM_save_path = args.res_CSV_savePath
    outData = open(GBM_save_path, "w")
    # outData.writelines("testingGroup,rmse_train,mape_train,r2_train,rmse_test,mape_test,r2_test,modelFile\n")
    outData.writelines(
        "testGroup,train_R2,train_MAE,train_RMSE,train_MAPE,train_large_MAPE,test_R2,test_MAE,test_RMSE,test_MAPE,test_large_MAPE\n"
    )
    all_y_label = []
    all_y_pred = []

    numGroups = 10  # group to test
    # only check one group
    if "timeSplit" in args.merged_data_dir:
        group_list = range(numGroups)
    else:
        group_list = [int(args.testGroup)]

    for n in group_list:
        print(f"test group is {n}")
        df_test = allData[allData["group"] == n]
        df_train = allData[allData["group"] != n]
        df_test = df_test.drop(["group"], axis=1)
        df_test.reset_index(drop=True, inplace=True)
        df_train = df_train.drop(["group"], axis=1)
        df_train.reset_index(drop=True, inplace=True)
        # get results
        res, temp_y_label, temp_y_pred = trainGBM(
            df_train, df_test, testGroup=n, bar=bar
        )
        all_y_label.extend(list(temp_y_label))
        all_y_pred.extend(list(temp_y_pred))
        # rmse_train, mape_train, r2_train, rmse_test, mape_test, r2_test, filename = res
        row = [str(n)] + [str(i) for i in res]
        row = ",".join(row)
        outData.writelines(row + "\n")

    # record all group predict results
    all_y_label = np.array(all_y_label)
    all_y_pred = np.array(all_y_pred)
    all_y_label_large_index = all_y_label >= bar
    all_test_MAE = "%.4f" % eval_on_batch_mae(all_y_pred, all_y_label)
    all_test_RMSE = "%.4f" % eval_on_batch_rmse(all_y_pred, all_y_label)
    all_test_R2 = "%.4f" % r2_score(all_y_label, all_y_pred)
    all_test_MAPE = "%.4f" % eval_on_batch_mape(all_y_pred, all_y_label)
    all_test_large_MAPE = "%.4f" % eval_on_batch_mape(
        all_y_pred[all_y_label_large_index], all_y_label[all_y_label_large_index]
    )
    row = [
        "allGroup",
        "-",
        "-",
        "-",
        "-",
        "-",
        str(all_test_R2),
        str(all_test_MAE),
        str(all_test_RMSE),
        str(all_test_MAPE),
        str(all_test_large_MAPE),
    ]
    row = ",".join(row)
    outData.writelines(row + "\n")

    outData.close()


def plotFigs(lgb, gbm, evals_result_dict, testGroup):
    if not os.path.exists(f"{args.saveDir}/Loss_plots"):
        os.makedirs(f"{args.saveDir}/Loss_plots")
    # plot figures
    print("Plot eval results during training process...")
    ax = lgb.plot_metric(evals_result_dict, metric="l1")
    plt.title("L1 loss")
    plt.savefig(f"{args.saveDir}/Loss_plots/train_l1_G{testGroup}.png")
    plt.close()
    ax = lgb.plot_metric(evals_result_dict, metric="huber")
    plt.title("Huber Loss")
    plt.savefig(f"{args.saveDir}/Loss_plots/train_huber_G{testGroup}.png")
    plt.close()
    print("Plot feature importance...")
    ax = lgb.plot_importance(gbm, figsize=(10, 25))
    plt.title("Feature importance")
    plt.savefig(f"{args.saveDir}/Loss_plots/feature_importances_G{testGroup}.png")
    plt.close()


def getMonitorR2(df, ordered_monitors, monitor_col="monitors"):
    R2_dict = {}
    for monitor in ordered_monitors:
        temp_df = df[df[monitor_col] == monitor]
        R2_dict[monitor] = r2_score(temp_df["PM2.5"], temp_df["pred_PM2.5"])
    return R2_dict


def plotR2Scatters(moniter_df, testGroup):
    # plot all data
    y_label = moniter_df["PM2.5"]
    y_pred = moniter_df["pred_PM2.5"]
    all_r2 = r2_score(y_label, y_pred)
    plt.figure(figsize=(8, 7))
    ax = plt.subplot(1, 1, 1)
    min_val = min(y_label.min(), y_pred.min())
    if min_val < 0:
        print("There are pred value < 0 !!!!")
        min_val = 0
    max_val = max(y_label.max(), y_pred.max())
    max_val = max(max_val, 85)
    plt.scatter(y_label, y_pred, s=4, alpha=0.25, cmap=plt.get_cmap("jet"))
    plt.plot(y_label, y_label, color="red", alpha=0.2)
    ax.text(20, 80, "R2:$%.3f$" % all_r2, size=16, ha="center", va="center")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Prediction")
    plt.tight_layout()
    if not os.path.exists(f"{args.saveDir}/R2_plots"):
        os.makedirs(f"{args.saveDir}/R2_plots")
    plt.savefig(f"{args.saveDir}/R2_plots/prediction_G{testGroup}_test.png", dpi=150)
    plt.close()
    # plot grid data scatter plot
    fig = plt.figure(figsize=(14, 14))
    sup_ax = plt.subplot(1, 1, 1)
    ordered_monitors = sorted(set(moniter_df["monitors"]))
    g = sns.FacetGrid(
        moniter_df, col="monitors", col_order=ordered_monitors, col_wrap=4
    )
    kws = dict(s=20, linewidth=0.5, edgecolor=None, alpha=0.3)
    map_g = (
        g.map(plt.scatter, "PM2.5", "pred_PM2.5", color="g", **kws)
        .set(xlim=(0, 100), ylim=(0, 100))
        .fig.subplots_adjust(wspace=0.25, hspace=0.25)
    )

    # add text in subplot
    R2_dict = getMonitorR2(moniter_df, ordered_monitors, monitor_col="monitors")
    for index, ax in g.axes_dict.items():
        ax.text(
            25, 50, "R2: $%.3f$" % R2_dict[index], size=12, ha="center", va="center"
        )

    fig.set_figheight(3)
    fig.set_figwidth(3)
    sup_ax.set_xlabel("Observation")
    sup_ax.set_ylabel("Prediction")
    plt.savefig(
        f"{args.saveDir}/R2_plots/prediction_G{testGroup}_test_grid.png", dpi=150
    )
    plt.close()


def main():
    if not os.path.exists(args.saveDir):
        os.makedirs(args.saveDir)
    trainGBM_CV_groupYears()


if __name__ == "__main__":
    main()

"""
python model_GBM_pretrainMergedData_allGroups.py --merged_data_dir preTrain_shapeNoDrop_allGroups --appendix reImplement --testGroup 0
"""
