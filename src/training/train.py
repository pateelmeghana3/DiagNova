import torch
import torch.nn as nn
import torch.optim as optim
from src.models.cnn_model import ChestCNN
from src.preprocessing.preprocess import get_dataloaders


# ----------------------------
# Device Selection
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)






# ----------------------------
# Training Function
# ----------------------------

def train_model():

    # Load Data

    train_loader, val_loader, test_loader = get_dataloaders()



    # Create Model

    model = ChestCNN().to(device)



    # Loss Function

    criterion = nn.CrossEntropyLoss()



    # Optimizer

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )



    # Epochs

    epochs = 10



    best_accuracy = 0



    for epoch in range(epochs):


        print(
            f"\nEpoch {epoch+1}/{epochs}"
        )


        # --------------------
        # Training
        # --------------------

        model.train()


        total_loss = 0

        correct = 0

        total = 0



        for images, labels in train_loader:


            images = images.to(device)

            labels = labels.to(device)



            # Forward pass

            outputs = model(images)



            # Calculate loss

            loss = criterion(
                outputs,
                labels
            )



            # Remove old gradients

            optimizer.zero_grad()



            # Backpropagation

            loss.backward()



            # Update weights

            optimizer.step()



            total_loss += loss.item()



            # Accuracy

            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()



        train_accuracy = (
            correct / total
        ) * 100



        print(
            "Training Loss:",
            total_loss / len(train_loader)
        )


        print(
            "Training Accuracy:",
            train_accuracy
        )



        # --------------------
        # Validation
        # --------------------

        model.eval()


        correct = 0

        total = 0



        with torch.no_grad():


            for images, labels in val_loader:


                images = images.to(device)

                labels = labels.to(device)



                outputs = model(images)



                _, predicted = torch.max(
                    outputs,
                    1
                )



                total += labels.size(0)


                correct += (
                    predicted == labels
                ).sum().item()



        val_accuracy = (
            correct / total
        ) * 100



        print(
            "Validation Accuracy:",
            val_accuracy
        )



        # --------------------
        # Save Best Model
        # --------------------

        if val_accuracy > best_accuracy:


            best_accuracy = val_accuracy


            torch.save(
                model.state_dict(),
                "models/saved_models/chest_cnn.pth"
            )


            print(
                "Best Model Saved ✅"
            )



    print(
        "\nTraining Completed!"
    )


    print(
        "Best Validation Accuracy:",
        best_accuracy
    )





# ----------------------------
# Run Training
# ----------------------------

if __name__ == "__main__":

    train_model()