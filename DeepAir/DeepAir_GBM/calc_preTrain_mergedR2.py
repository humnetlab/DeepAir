import numpy as np
import pandas as pd


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


model_suffix = "preTrain_shapeNoDrop_preProcess_0"  # GBM output
merged_df = None
for testGroup in range(10):
    pred_path = (
        f"./GBM/{model_suffix}/{testGroup}/pred_results/prediction_G{testGroup}.csv"
    )
    temp_df = pd.read_csv(pred_path)
    temp_df["testGroup"] = testGroup
    if testGroup == 0:
        merged_df = temp_df.copy()
    else:
        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)
merged_df.to_csv(f"./GBM/{model_suffix}/merged_predResults.csv", index=0)
