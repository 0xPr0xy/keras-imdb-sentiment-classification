from classes import *

dataset = DatasetIMDB()
dataset.configure()

trained_model = TrainedModel()

(trained_model
    .retrain(dataset)
    .load_from_json()
    .compile()
    .load_weights()
    .evaluate(dataset)
    .predict(dataset)
)
