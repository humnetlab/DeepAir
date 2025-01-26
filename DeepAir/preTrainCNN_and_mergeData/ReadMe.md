# preTrainCNN 

- The main data for training the model is read from `./data/DeepAir/STData`, `./data/DeepAir/STData/Geo`, and `./data/DeepAir/STData/NLCD`.
- Pay attention to the data reading operations in `load_static_data.py`, which records the storage paths for some static data.
- During model training, there is a `ensembleTest` step, where predictions are repeated multiple times, and the average result is computed for evaluation.
- The results are stored in the folder `ElevaLand.preTrain.shapeNoDrop`:
    - `predResults` contains prediction results for different Groups.
    - `saved_earlyStop_weights_yearsGx` stores the model parameters saved when different Group data is used as the test set. These parameters are also used in later work for encoding LandCover and Elevation.

The model parameters during training are saved based on different Groups serving as the test data.


# CNN_embed_and_merge_data 

Train the CNN model and encode data for subsequent GBM model training.  
This process relies on the pre-trained models stored in the `ElevaLand.preTrain.shapeNoDrop` folder under `preTrainCNN`.

Different Groups represent the use of different Groups as the test set during pre-training.

The output data is stored in the `mergedData` path.