import torch
from tqdm import tqdm

from training_utils import TrainingUtilities


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

    def train_model(self, verbose=0):
        model, optimizer, criterion, epochs, dataloaders = self.model, self.optimizer, self.criterion, self.epochs, self.dataloaders
        device = self.DEVICE

        training_epoch = 0
        epoch = 0
        if self.is_continue:

            if verbose > 0:
                print(f"Continuing from checkpoint {self.checkpoint}")

            epoch, model, optimizer = TrainingUtilities.load_checkpoint(model, optimizer, self.checkpoint, self.scheduled, verbose)

        for training_epoch in range(epoch, epochs):

            train_losses = []
            val_losses = []
            val_accuracies = []
            avg_train_loss = 0
            avg_val_loss = 0
            val_accuracy = 0
            train_accuracy = 0

            for phase in ['train', 'val']:
                if phase == 'train':
                    train_loader = dataloaders['train']
                    model.train()
                    train_loss = 0
                    correct_train = 0
                    total_train = 0
                    for batch in tqdm(train_loader, desc=f"Epoch {training_epoch + 1}/5 - Training"):
                        video = batch['video'].to(device)  # (Batch, Frames, C, H, W)
                        target = batch['target'].to(device)

                        optimizer.zero_grad()
                        outputs = model(video)
                        loss = criterion(outputs, target)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        correct_train += (predicted == target).sum().item()
                        total_train += target.size(0)

                    avg_train_loss = train_loss / len(train_loader)
                    train_accuracy = correct_train / total_train
                else:
                    val_loader = dataloaders['val']
                    avg_val_loss, val_accuracy = self.evaluate(val_loader, training_epoch)

            print(
                f"Epoch [{epoch + 1}/5], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")        #Test model
        self.test(dataloaders['test'])

    def test(self, test_loader):
        model, criterion,  device = self.model, self.criterion, self.DEVICE
        model.eval()
        test_loss = 0

        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                video = batch['video'].to(device)
                target = batch['target'].to(device)

                outputs = model(video)
                loss = criterion(outputs, target)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == target).sum().item()
                total_test += target.size(0)

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct_test / total_test
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        TrainingUtilities.save_model(model,'final_', self.save_folder)

    def evaluate(self,val_loader,epoch, verbose=0):
        model, criterion, device = self.model, self.criterion, self.DEVICE
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/5 - Validation"):
                video = batch['video'].to(device)
                target = batch['target'].to(device)

                outputs = model(video)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == target).sum().item()
                total_val += target.size(0)
        val_accuracy = correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss, val_accuracy
