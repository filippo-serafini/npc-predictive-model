# Npc-predictive-model
Repo for the project activity on Node Power Consumption ML model for HPC jobs. It contains both traditional and federated approaches.

## Repository structure
The 'Traditional' folder contains the centralized approach to train the network. In particular:
- [FDATAReader.py](Traditional/FDATAReader.py) and [PM100Reader.py](Traditional/PM100Reader.py) allows you to read the parquet files containing the jobs features.
- [NPCdataset.py](Traditional/NPCdataset.py) script contains the dataset class to wrap the jobs for training, validation and testing.
- [NPCnetwork.py](Traditional/NPCnetwork.py) script reports the model of the nn and all the training, validation and test functions.

The 'Federated Learning' folder keeps the federated approach developed using Flower.ai framework.
- [pyproject.toml](FederatedLearning/my-app/pyproject.toml) is the configuration file to specify runtime settings.
- [client_app.py](FederatedLearning/my-app/my_app/client_app.py) contains the logic for the client app.
- [server_app.py](FederatedLearning/my-app/my_app/server_app.py) contains the logic for the server app.
- [task.py](FederatedLearning/my-app/my_app/task.py) keeps all the functions used by both server and clients for training, validate and test.
- [NPCdataset.py](FederatedLearning/my-app/my_app/NPCdataset.py) script contains the dataset class to wrap the jobs for training, validation and testing.
- [NPCnetwork.py](FederatedLearning/my-app/my_app/NPCnetwork.py) script reports the model of the nn.

The 'Datasets' folder contains all the datasets for training, validation and test obtained aggregating PM100 and FDATA jobs.
