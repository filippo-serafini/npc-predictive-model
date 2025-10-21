import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from my_app.NPCnetwork import NpcNetwork
from my_app.task import load_data
from my_app.task import validate as test_fn
from my_app.task import train as train_fn

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights from Message msg
    model = NpcNetwork()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data for the specific client:
    # partition_id or client_id, number of partitions or number of clients
    # We're only interested in the train dataloader here
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, _ = load_data(partition_id, num_partitions)

    # Call the training function
    train_loss = train_fn(
        net=model,
        trainloader=train_loader,
        local_epochs=context.run_config["local-epochs"],
        lr=msg.content["config"]["lr"],
        device=device,
    )

    # Construct and return reply Message:
    # num-examples is required by FedAvg to weight client's parameters
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(train_loader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = NpcNetwork()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    # We're only interested in the validation dataloader here
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, val_loader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss = test_fn(
        model,
        val_loader,
        device,
    )

    # Construct and return reply Message:
    # num-examples is required by FedAvg to weight client's parameters
    metrics = {
        "eval_loss": eval_loss,
        "num-examples": len(val_loader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)