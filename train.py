from classes import *

dataset = DatasetIMDB()     # initialize dataset for training and testing, offers method for sentence reconstruction

trained_model = TrainedModel()

(trained_model
    .set_dataset(dataset)   # set the dataset on the trained model, this can be used to retrain or evaluate / make predictions
    .retrain()              # retrain the model (exports the model & weights and currently does not do evaluation or prediction)
    .load_from_json()       # load the model from json
    .compile()              # compile the loaded model
    .load_weights()         # load pretrained weights in the model
    .evaluate()             # evaluate the dataset
    .predict()              # make predictions and print expected prediction, actual prediction and sentence on which the prediction was made
)
