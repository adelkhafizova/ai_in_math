import matplotlib.pyplot as plt


def plot_epochs_avg_loss(avg_loss_arr,
                         file_name):
    epochs = list(range(1, len(avg_loss_arr) + 1))

    plt.figure()
    plt.plot(epochs, avg_loss_arr)

    plt.xlim(left=0)  # x‑axis from 0
    plt.ylim(bottom=0)  # y‑axis from 0

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')

    plt.tight_layout()

    plt.savefig(file_name)
    plt.close()


def plot_word_power_range(word,
                          power_arr,
                          file_name):
    """
    Plot the power range at each step for a given word.
    word: string of actions (e.g. 'AbBAa...')
    power_arr: list of integers, one per step
    output_path: file path to save the plot
    """
    steps = list(range(1, len(power_arr) + 1))
    plt.figure()
    plt.plot(steps, power_arr)
    plt.xlabel('Step')
    plt.ylabel('Power Range')
    plt.title(f'Word: {word}')
    plt.tight_layout()

    plt.savefig(file_name)
    plt.close()