import torch

from training_utils import TrainingUtilities
import time
from sklearn.metrics import f1_score


class ModelTrainer:
    def __init__(self, model, optimizer, scheduled, criterion, epochs, dataloaders, device, save_folder,
                 is_continue=False, checkpoint=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduled = scheduled
        self.criterion = criterion
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.DEVICE = device
        self.save_folder = save_folder
        self.is_continue = is_continue
        self.checkpoint = checkpoint
        self.scaler = torch.amp.GradScaler()

    def train_model(self, verbose=0):
        model, optimizer, criterion, epochs, dataloaders = self.model, self.optimizer, self.criterion, self.epochs, self.dataloaders
        device = self.DEVICE
        scaler = self.scaler
        training_epoch = 0
        epoch = 0
        if self.is_continue:

            if verbose > 0:
                print(f"Continuing from checkpoint {self.checkpoint}")

            epoch, model, optimizer = TrainingUtilities.load_checkpoint(model, optimizer, self.checkpoint, self.scheduled, verbose)

        train_losses = []
        val_losses = []
        test_losses = []

        val_accuracies = []
        train_accuracies = []
        test_accuracies = []

        for training_epoch in range(epoch, epochs):
            print(f"\nTraining epoch {training_epoch+1}")


            avg_train_loss = 0
            avg_val_loss = 0
            val_accuracy = 0
            train_accuracy = 0
            train_time = 0
            val_time = 0
            val_f1 = 0
            for phase in ['train', 'val']:
                if phase == 'train':
                    start_time = time.time()
                    # print("Training phase.....")
                    train_loader = dataloaders['train']
                    model.train()
                    train_loss = 0
                    correct_train = 0
                    total_train = 0
                    for batch in train_loader:
                        video = batch['video'].to(device)  # (Batch, Frames, C, H, W)
                        target = batch['target'].to(device)

                        optimizer.zero_grad()
                        with torch.amp.autocast('cuda'):
                            outputs = model(video)
                            loss = criterion(outputs, target)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        train_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        correct_train += (predicted == target).sum().item()
                        total_train += target.size(0)

                    avg_train_loss = train_loss / len(train_loader)
                    train_accuracy = correct_train / total_train
                    end_time = time.time()
                    train_time = end_time - start_time
                    formatted_time = time.strftime("%Mmins %Ssecs", time.gmtime(train_time))
                    print(f"Training completed in {formatted_time}")
                else:
                    # print("Validation phase.....")
                    val_loader = dataloaders['val']
                    start_time = time.time()
                    avg_val_loss, val_accuracy, val_f1 = self.evaluate(val_loader, training_epoch)
                    end_time = time.time()
                    val_time = end_time - start_time
                    formatted_time = time.strftime("%Mmins %Ssecs", time.gmtime(val_time))
                    print(f"Validation completed in {formatted_time}")
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            print(
                f"Epoch [{training_epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}"
            )
            start_time = time.time()
            avg_test_loss, test_accuracy = self.test(dataloaders['test'], training_epoch+1)
            end_time = time.time()
            test_time = end_time - start_time
            formatted_time = time.strftime("%Mmins %Ssecs", time.gmtime(test_time))
            print(f"Testing completed in {formatted_time}")
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)

        return train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies

    def test(self, test_loader, epoch):
        model, criterion, device = self.model, self.criterion, self.DEVICE
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for batch in test_loader:
                    video = batch['video'].to(device)
                    target = batch['target'].to(device)

                    outputs = model(video)
                    loss = criterion(outputs, target)
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct_test += (predicted == target).sum().item()
                    total_test += target.size(0)

                    # Store predictions and true labels for F1-score
                    y_true.extend(target.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct_test / total_test
        test_f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")
        TrainingUtilities.save_model(model, f'model_epoch{epoch}-acc{test_accuracy:.2f}', self.save_folder)
        return avg_test_loss, test_accuracy

    def evaluate(self, val_loader, epoch, verbose=0):
        model, criterion, device = self.model, self.criterion, self.DEVICE
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for batch in val_loader:
                    video = batch['video'].to(device)
                    target = batch['target'].to(device)

                    outputs = model(video)
                    loss = criterion(outputs, target)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == target).sum().item()
                    total_val += target.size(0)

                    # Store predictions and true labels for F1-score
                    y_true.extend(target.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

        val_accuracy = correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(y_true, y_pred, average='weighted')

        return avg_val_loss, val_accuracy, val_f1

