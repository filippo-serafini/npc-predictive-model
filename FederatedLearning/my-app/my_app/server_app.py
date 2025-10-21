import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from my_app.NPCnetwork import NpcNetwork
from my_app.task import central_evaluate, test_final_model, initialize_transforms

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config (pyproject.toml)
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = NpcNetwork()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize transform and target_transform
    trsf, target_trsf = initialize_transforms()

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_train
    )

    # Start strategy, run FedAvg for `num_rounds`
    # Validation also on server-side with central_evaluate fun in task.py
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=central_evaluate,
    )

    # Final testing
    final_model = NpcNetwork()
    final_model.load_state_dict(result.arrays.to_torch_state_dict())

    test_final_model(final_model, trsf, target_trsf)

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")