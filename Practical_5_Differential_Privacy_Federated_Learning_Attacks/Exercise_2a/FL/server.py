import flwr as fl


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy:": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=fl.server.strategy.FedAvg(
            min_available_clients=6,
            evaluate_metrics_aggregation_fn=weighted_average
        )
    )
