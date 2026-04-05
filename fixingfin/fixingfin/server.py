import flwr as fl

def weighted_average(metrics):
    total = sum(num_examples for num_examples, _ in metrics)

    return {
        "accuracy": sum(n * m["accuracy"] for n, m in metrics) / total,
        "precision": sum(n * m["precision"] for n, m in metrics) / total,
        "recall": sum(n * m["recall"] for n, m in metrics) / total,
        "f1": sum(n * m["f1"] for n, m in metrics) / total,
    }

strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)