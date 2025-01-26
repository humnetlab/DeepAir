"""Update: change temporal meteorological data to spatio-tempral"""

import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import DeepAir.preTrainCNN_and_mergeData.utils as utils
from DeepAir.preTrainCNN_and_mergeData.utils import *

spatialWindowSize = config["spatialWindowSize"]
temporalWindowSize = config["temporalWindowSize"]

if os.path.exists("./data/DeepAir/"):
    dataPath = "./data/DeepAir/"

CA_boundary = [-124.48, -114.13, 32.53, 42.01]
metFeatures = [
    "TEMP",
    "Humidity",
    "PRESS",
    "uWIND",
    "vWIND",
    "Evaporation",
    "Precipitation",
]

normalizedFeatures = []
normalizedFeatures += ["Lon", "Lat"]
normalizedFeatures += ["Elevation", "EVI", "AOD"]  # "Elevation", "AOD"
normalizedFeatures += ["Emission", "DistanceToRoads"]  # "Emission"


def get_situFlag(case_label):
    """if flag is True, then only use situData"""
    if case_label == "met3DEEESA":
        situFlag = {
            "AOD": False,
            "Emission": False,
            "Elevation": False,
            "EVI": False,
            "Met": False,
            "SR": False,
            "Fire": True,
        }
    elif case_label == "met3DEES":
        situFlag = {
            "AOD": True,
            "Emission": True,
            "Elevation": False,
            "EVI": False,
            "Met": False,
            "SR": False,
            "Fire": True,
        }
    elif case_label == "LandCover":
        situFlag = {
            "AOD": True,
            "Emission": True,
            "Elevation": True,
            "EVI": True,
            "Met": True,
            "SR": True,
            "Fire": True,
        }
    elif case_label == "Elevation":
        situFlag = {
            "AOD": True,
            "Emission": True,
            "Elevation": False,
            "EVI": True,
            "Met": True,
            "SR": True,
            "Fire": True,
        }
    elif case_label == "ElevaLand":
        situFlag = {
            "AOD": True,
            "Emission": True,
            "Elevation": False,
            "EVI": True,
            "Met": True,
            "SR": True,
            "Fire": True,
        }
    elif case_label == "EELand":
        situFlag = {
            "AOD": True,
            "Emission": False,
            "Elevation": False,
            "EVI": True,
            "Met": True,
            "SR": True,
            "Fire": True,
        }
    elif case_label == "EEESLand":
        situFlag = {
            "AOD": True,
            "Emission": False,
            "Elevation": False,
            "EVI": False,
            "Met": True,
            "SR": False,
            "Fire": True,
        }
    elif case_label == "met3DEEESALand":
        situFlag = {
            "AOD": False,
            "Emission": False,
            "Elevation": False,
            "EVI": False,
            "Met": False,
            "SR": False,
            "Fire": True,
        }
    elif case_label == "ELS":
        situFlag = {
            "AOD": True,
            "Emission": True,
            "Elevation": False,
            "EVI": True,
            "Met": True,
            "SR": False,
            "Fire": True,
        }
    elif case_label == "ElevaLandFire":
        situFlag = {
            "AOD": True,
            "Emission": True,
            "Elevation": False,
            "EVI": True,
            "Met": True,
            "SR": True,
            "Fire": True,
        }
    else:
        print("\n\nIllegal case_label!\n")
        exit(0)
    return situFlag


class MySet(Dataset):
    def __init__(self, args, input_file):
        self.content = []
        # load elevation data
        inData = open(dataPath + "STData/monitorElevation.json", "r")
        for row in inData:
            elevationDict = json.loads(row)
        inData.close()

        # load emission data, yearly data per row
        inData = open(dataPath + "STData/monitorEmission.json", "r")
        emisionDict = {}
        for row in inData:
            row_dict = json.loads(row)
            emisionDict[row_dict["year"]] = row_dict
        inData.close()

        # load the land cover histgram
        gridLandCover = pickle.load(
            open(dataPath + "NLCD/gridLandCovers_hist.pkl", "rb")
        )

        # load distance to roads
        inData = open(dataPath + "Geo/gridsDistanceToRoads.csv", "r")
        inData.readline()
        distanceToRoads = {}
        for row in inData:
            row = row.rstrip().split(",")
            lon = float(row[1])
            lat = float(row[2])
            dist = float(row[3]) / 1000.0  # km
            distanceToRoads[(lon, lat)] = np.round(dist, 4)
        inData.close()

        # load the training data
        inData = open(input_file, "r")
        # inData.readline()

        count = 0
        count_kept = 0

        situFlag = get_situFlag(args.case_label)

        for row in inData:
            count += 1
            try:
                row = json.loads(row)
            except:
                print("Error in load json")
                continue

            # attributes
            yearDay = row["yearDay"]
            year = int(yearDay[:4])
            yearID = year - 2006
            dayIdx = int(yearDay[4:])
            month = row["month"]
            weekday = row["weekday"]
            basinID = row["basinID"]
            Lon = row["Lon"]
            Lat = row["Lat"]
            loc_str = str(Lon) + "_" + str(Lat)

            DistanceToRoads = distanceToRoads[(Lon, Lat)]
            PM25 = row["PM2.5"]

            # attibutes
            row_dict = {}
            row_dict["yearDay"] = int(yearDay)
            row_dict["yearID"] = yearID
            row_dict["dayIdx"] = dayIdx
            row_dict["month"] = month
            row_dict["weekday"] = weekday
            row_dict["Lon"] = Lon
            row_dict["Lat"] = Lat
            row_dict["basinID"] = basinID
            row_dict["DistanceToRoads"] = DistanceToRoads
            row_dict["PM2.5"] = PM25
            row_dict["intercept"] = 1.0

            # situ data
            # row["AOD"] 7 length list,  element length is 2601
            if situFlag["AOD"]:
                row_dict["AOD"] = row["AOD"][-1][1300]
            else:
                row_dict["AOD"] = np.asarray(row["AOD"][-1]).reshape(
                    (spatialWindowSize, spatialWindowSize)
                )

            # SR_b* 2601 length list
            SR_vec = []
            for b in range(7):
                srKey = "SR_b" + str(b)
                if situFlag["SR"]:
                    SR_vec.append(row[srKey][1300])
                else:
                    SR_vec.append(
                        np.asarray(row[srKey]).reshape(
                            spatialWindowSize, spatialWindowSize
                        )
                    )

            if situFlag["SR"] == False:
                SR_vec = SR_vec[-1]

            row_dict["SR"] = SR_vec

            if situFlag["Elevation"]:
                row_dict["Elevation"] = elevationDict[loc_str][1300]
            else:
                row_dict["Elevation"] = 0  # will be replaced in train
                if "Elevation" in normalizedFeatures:
                    normalizedFeatures.remove("Elevation")

            situEmission = np.asarray(emisionDict[year][loc_str]) / 365.0 / 24.0
            situEmission = np.round(situEmission, 4)
            if situFlag["Emission"]:
                row_dict["Emission"] = situEmission[1300]
            else:
                row_dict["Emission"] = 0  # will be replaced in train

            # row["EVI"] 2601 length list
            if situFlag["EVI"]:
                row_dict["EVI"] = row["EVI"][1300]
            else:
                row_dict["EVI"] = np.asarray(row["EVI"]).reshape(
                    (spatialWindowSize, spatialWindowSize)
                )

            row_dict["LandCover"] = gridLandCover[(Lon, Lat)]

            try:
                fire = np.asarray(row["Fire"][-1]).reshape(
                    spatialWindowSize, spatialWindowSize
                )
            except:
                fire = np.zeros(shape=(spatialWindowSize, spatialWindowSize))

            if situFlag["Fire"]:
                row_dict["Fire"] = normalize(np.sum(fire), "Fire")
            else:
                row_dict["Fire"] = fire  # binary image

            # 3D Meteorological data
            # normalize meteorological data
            Met3D = []  # only the current day
            # row[met] = [[d1], ...,[d7]]
            WindowSize_met = 5
            for met in metFeatures:
                tmp = np.array(row[met][-1])
                tmp = np.nan_to_num(tmp)
                tmp = normalize(tmp, met)
                if situFlag["Met"]:
                    Met3D.append(tmp[12])  # center grid value
                else:
                    metMatrix = tmp.reshape(
                        (WindowSize_met, WindowSize_met)
                    )  # spatial value
                    metMatrix = metMatrix.tolist()
                    Met3D.append(metMatrix)

            row_dict["Met"] = Met3D  # 7x5x5

            if (
                row_dict["PM2.5"] < 0.5
                or row_dict["PM2.5"] > 400
                or row_dict["EVI"] < 0
            ):
                continue
            for fea in normalizedFeatures:
                row_dict[fea] = utils.normalize(row_dict[fea], fea)

            self.content.append(row_dict)
            count_kept += 1

        inData.close()
        # disorder the content
        random.shuffle(self.content)
        print("# samples loaded: %d / %d" % (count_kept, count))
        self.lengths = list(map(lambda x: 1, self.content))

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(list(self.content))


def collate_fn(data):
    stat_attrs = ["PM2.5"]
    long_attrs = ["yearDay", "yearID", "dayIdx", "month", "weekday", "basinID"]
    info_attrs = ["Lon", "Lat", "Fire", "DistanceToRoads", "intercept"]
    vector_attrs = ["LandCover", "SR"]  # "gridLocVec", "NeighborPM2.5"
    vector2D_attrs = [
        "Elevation",
        "AOD",
        "EVI",
        "Emission",
        "Met",
    ]

    attr = {}

    for key in stat_attrs:
        attr[key] = torch.FloatTensor([item[key] for item in data])
    for key in long_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])
    for key in info_attrs:
        attr[key] = torch.FloatTensor([item[key] for item in data])
    for key in vector_attrs:
        attr[key] = torch.FloatTensor([np.asarray(item[key]) for item in data])

    for key in vector2D_attrs:
        img_list = []
        for item in data:
            tmp = item[key]  # 3D
            img_list.append(tmp)
        imgs = np.asarray(img_list)  # 4D
        imgs = torch.from_numpy(imgs).float()
        attr[key] = imgs
    return attr


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))

    def __iter__(self):
        """
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        """
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size : (i + 1) * chunk_size]
            partial_indices.sort(key=lambda x: self.lengths[x], reverse=True)
            self.indices[i * chunk_size : (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size : (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


class BatchSampler_fromData:
    def __init__(self, dataset_numSamples, dataset_lengths, batch_size, shufData=True):
        self.count = dataset_numSamples  # data sample number
        self.batch_size = batch_size
        self.lengths = dataset_lengths  # [1, 1, 1, ... , 1]
        self.indices = list(range(self.count))
        self.shufData = shufData  # shuffle or not

    def __iter__(self):
        """
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        """
        if self.shufData == True:
            np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size : (i + 1) * chunk_size]
            partial_indices.sort(key=lambda x: self.lengths[x], reverse=True)
            self.indices[i * chunk_size : (i + 1) * chunk_size] = partial_indices
        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size
        for i in range(batches):
            yield self.indices[i * self.batch_size : (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def get_loader(input_file, batch_size):
    dataset = MySet(input_file=input_file)

    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=lambda x: collate_fn(x),
        num_workers=4,
        batch_sampler=batch_sampler,
        pin_memory=True,
    )

    return data_loader


def get_loader_fromData(input_data, batch_size, shufData=True):
    dataset_lengths = list(map(lambda x: 1, input_data))
    dataset_numSamples = len(input_data)

    batch_sampler = BatchSampler_fromData(
        dataset_numSamples, dataset_lengths, batch_size, shufData=shufData
    )

    data_loader = DataLoader(
        dataset=input_data,
        batch_size=1,
        collate_fn=lambda x: collate_fn(x),
        num_workers=0,
        batch_sampler=batch_sampler,
        pin_memory=True,
    )

    return data_loader
