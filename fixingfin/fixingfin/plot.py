import matplotlib.pyplot as plt

def plot_metrics(history):
    rounds = range(1, len(history["loss"]) + 1)

    plt.figure()
    plt.plot(rounds, history["loss"])
    plt.title("Loss vs Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.plot(rounds, history["accuracy"])
    plt.title("Accuracy vs Rounds")
    plt.show()

    plt.figure()
    plt.plot(rounds, history["recall"])
    plt.title("Recall vs Rounds")
    plt.show()

    plt.figure()
    plt.plot(rounds, history["f1"])
    plt.title("F1 Score vs Rounds")
    plt.show()