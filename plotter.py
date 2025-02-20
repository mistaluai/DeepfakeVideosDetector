class Plotter:
    def __init__(self, save_dir="plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_loss(self, train_loss, val_loss, test_loss=None, filename="loss_plot.png"):
        plt.figure(figsize=(8, 6))
        plt.plot(train_loss, label="Train Loss", marker='o')
        plt.plot(val_loss, label="Validation Loss", marker='o')
        if test_loss is not None:
            plt.plot(test_loss, label="Test Loss", marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training, Validation, and Test Loss")
        plt.legend()
        plt.grid()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Loss plot saved to {save_path}")

    def plot_accuracy(self, train_acc, val_acc, test_acc=None, filename="accuracy_plot.png"):
        plt.figure(figsize=(8, 6))
        plt.plot(train_acc, label="Train Accuracy", marker='o')
        plt.plot(val_acc, label="Validation Accuracy", marker='o')
        if test_acc is not None:
            plt.plot(test_acc, label="Test Accuracy", marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training, Validation, and Test Accuracy")
        plt.legend()
        plt.grid()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Accuracy plot saved to {save_path}")