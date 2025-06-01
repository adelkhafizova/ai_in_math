import matplotlib.pyplot as plt


def plot_epochs_avg_loss(avg_loss_arr):
    epochs = list(range(1, len(avg_loss_arr) + 1))

    plt.figure()
    plt.plot(epochs, avg_loss_arr)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')

    plt.tight_layout()

    plt.savefig('avg_loss.png')
    plt.close()