""" Load static data in space to save memory """

import json
import pickle

import numpy as np
import torch
from PIL import Image

import DeepAir.preTrainCNN_and_mergeData.utils as utils

spatialWindowSize = utils.config["spatialWindowSize"]
temporalWindowSize = utils.config["temporalWindowSize"]


dataPath = "./data/DeepAir/"


def readImage(monitor_str):
    monitor_str = "_".join(monitor_str) + "_buf1"  # the situ image
    imgPath = dataPath + "GridImgs/" + monitor_str + ".png"
    im = Image.open(imgPath).convert("RGB")
    im = im.resize((200, 200), Image.NEAREST)
    im = np.array(im)
    im = np.rollaxis(im, 2, 0)
    im = np.divide(im, 255.0)
    return im


# load basemap images for all monitors
def loadBaseMapImgs():
    # location to monitor ID (state, county, site)
    monitorLocToID_CA = pickle.load(open(dataPath + "Geo/monitorLocToID_CA.pkl", "rb"))
    # load map image
    baseMapDict = {}
    for loc in monitorLocToID_CA:
        monitorID = monitorLocToID_CA[loc]
        monitor_str = [str(i) for i in monitorID]
        im = readImage(monitor_str)
        baseMapDict[loc] = im
    return baseMapDict


# add basemap images to attr
def addBaseMapsToBatchData(attr, baseMapDict):
    Lon_list = list(attr["Lon"])
    Lat_list = list(attr["Lat"])
    imgs = []
    for l in range(len(Lon_list)):
        # Lon, Lat are normalized in data_loader
        Lon = utils.unnormalize(float(Lon_list[l]), "Lon")
        Lat = utils.unnormalize(float(Lat_list[l]), "Lat")
        Lon = np.round(Lon, 2)
        Lat = np.round(Lat, 2)
        im = baseMapDict[(Lon, Lat)]
        imgs.append(im)
    imgs = np.asarray(imgs)  # 4D
    imgs = torch.from_numpy(imgs).float()
    return imgs


# load elevation
def loadElevation():
    # load elevation data
    inData = open(dataPath + "STData/monitorElevation.json", "r")
    for row in inData:
        elevationDict = json.loads(row)
    inData.close()
    return elevationDict


# add elevation matrix to attr
def addElevationToBatchData(attr, elevationDict):
    Lon_list = list(attr["Lon"])
    Lat_list = list(attr["Lat"])
    elevation_list = []
    for l in range(len(Lon_list)):
        # Lon, Lat are normalized in data_loader
        Lon = utils.unnormalize(float(Lon_list[l]), "Lon")
        Lat = utils.unnormalize(float(Lat_list[l]), "Lat")
        Lon = np.round(Lon, 2)
        Lat = np.round(Lat, 2)
        loc_str = str(Lon) + "_" + str(Lat)
        elevation = np.asarray(elevationDict[loc_str]).reshape(
            (spatialWindowSize, spatialWindowSize)
        )
        elevation = utils.normalize(elevation, "Elevation")
        elevation_list.append(elevation)
    elevation_list = np.asarray(elevation_list)  #
    elevation_list = torch.from_numpy(elevation_list).float()
    return elevation_list


# load emission
def loadEmission():
    inData = open(dataPath + "STData/monitorEmission.json", "r")
    emisionDict = {}
    for row in inData:
        row_dict = json.loads(row)
        emisionDict[row_dict["year"]] = row_dict
    inData.close()
    return emisionDict


# add emission matrix to attr
def addEmissionToBatchData(attr, emisionDict):
    Lon_list = list(attr["Lon"])
    Lat_list = list(attr["Lat"])
    years = list(attr["yearID"])
    emission_list = []
    for l in range(len(Lon_list)):
        # Lon, Lat are normalized in data_loader
        Lon = utils.unnormalize(float(Lon_list[l]), "Lon")
        Lat = utils.unnormalize(float(Lat_list[l]), "Lat")
        Lon = np.round(Lon, 2)
        Lat = np.round(Lat, 2)
        loc_str = str(Lon) + "_" + str(Lat)
        year = years[l] + 2006
        year = int(year)

        emission = np.asarray(emisionDict[year][loc_str]) / 365.0 / 24.0
        emission = emission.reshape((spatialWindowSize, spatialWindowSize))
        emission = utils.normalize(emission, "Emission")
        emission = np.round(emission, 4)
        emission_list.append(emission)
    emission_list = np.asarray(emission_list)  #
    emission_list = torch.from_numpy(emission_list).float()
    return emission_list


# read land cover image
def readLandCoverImage(loc_str):
    imgPath = dataPath + "NLCD/gridNLCDimages/" + loc_str + ".png"
    im = Image.open(imgPath).convert("RGB")
    im = im.resize((100, 100), Image.NEAREST)
    im = np.array(im)
    im = np.rollaxis(im, 2, 0)
    im = np.divide(im, 255.0)
    return im


# load land cover images for all monitors
# sample: 12014-3693.png
def loadLandCoverImgs():
    # location to monitor ID (state, county, site)
    monitorLocToID_CA = pickle.load(open(dataPath + "Geo/monitorLocToID_CA.pkl", "rb"))
    # load map image
    landCoverImagesDict = {}
    for loc in monitorLocToID_CA:
        Lon, Lat = loc
        loc_str = str(int(-Lon * 100)) + "-" + str(int(Lat * 100))
        im = readLandCoverImage(loc_str)
        landCoverImagesDict[loc] = im
    return landCoverImagesDict


# add landcover images to attr
def addLandCoverToBatchData(attr, landCoverImagesDict):
    Lon_list = list(attr["Lon"])
    Lat_list = list(attr["Lat"])
    imgs = []
    for l in range(len(Lon_list)):
        # Lon, Lat are normalized in data_loader
        Lon = utils.unnormalize(float(Lon_list[l]), "Lon")
        Lat = utils.unnormalize(float(Lat_list[l]), "Lat")
        Lon = np.round(Lon, 2)
        Lat = np.round(Lat, 2)
        im = landCoverImagesDict[(Lon, Lat)]
        imgs.append(im)
    imgs = np.asarray(imgs)  # 4D
    imgs = torch.from_numpy(imgs).float()
    return imgs


def loadFireMatrix():
    # load gridFire matrix from pkl file
    # key-value gridFire_matrix[yearDay][(lon,lat)] = np.array(2601)
    with open(dataPath + "STData/monitorFireMatrix.pkl", "rb") as f:
        gridFire_matrix = pickle.load(f)
    return gridFire_matrix


def addFireMatrixToBatchData(attr, gridFire_matrix):
    yearDay_list = list(attr["yearDay"])
    Lon_list = list(attr["Lon"])
    Lat_list = list(attr["Lat"])
    assert len(yearDay_list) == len(Lon_list) and len(Lon_list) == len(Lat_list)
    gridFire_matrix_list = []
    for i in range(len(yearDay_list)):
        yearDay = str(yearDay_list[i].item())
        Lon = utils.unnormalize(float(Lon_list[i]), "Lon")
        Lat = utils.unnormalize(float(Lat_list[i]), "Lat")
        Lon = np.round(Lon, 2)
        Lat = np.round(Lat, 2)
        loc = (Lon, Lat)
        if yearDay in gridFire_matrix.keys():
            if loc in gridFire_matrix[yearDay].keys():
                loc_fire_matrix = np.asarray(gridFire_matrix[yearDay][loc]).reshape(
                    (spatialWindowSize, spatialWindowSize)
                )
            else:
                loc_fire_matrix = np.zeros((spatialWindowSize, spatialWindowSize))
        else:
            loc_fire_matrix = np.zeros((spatialWindowSize, spatialWindowSize))
        loc_fire_matrix = duplicateChannel(loc_fire_matrix)
        gridFire_matrix_list.append(loc_fire_matrix)
    gridFire_matrix_list = np.asarray(gridFire_matrix_list)
    gridFire_matrix_list = torch.from_numpy(gridFire_matrix_list).float()
    return gridFire_matrix_list


def duplicateChannel(singleChannel):
    """
    For Fire Data
    convert single channel image to multi-channel(3) image
    just duplicate values in single channel
    """
    imgMatrix = singleChannel * 255.0  # time 255.0 when dealing with fire image
    multi_channel_matrix = np.zeros([3, imgMatrix.shape[0], imgMatrix.shape[1]])
    for i in range(3):
        multi_channel_matrix[i, :, :] = imgMatrix
    return multi_channel_matrix
