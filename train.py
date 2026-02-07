import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_loaders
from models.protopnet import ProtoPNet
from utils.adaptive import ( build_feature_bank,refine_prototypes,update_prototypes)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader,test_loader =get_loaders(batch_size=64)
model=ProtoPNet(num_classes=10, num_prototypes=30).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(
    model.parameters(),
    lr=1e-3
)
def train_one_epoch(epoch):
    model.train()
    running_loss =0
    loop=tqdm(train_loader)
    for images,labels in loop:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        logits, similarity=model(images)
        loss=criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        loop.set_description(f"Ep {epoch}")
        loop.set_postfix(loss=loss.item())
    return running_loss/len(train_loader)
def main():
    epochs=10
    for epoch in range(epochs):
        loss=train_one_epoch(epoch)
        if epoch % 5 == 0 and epoch > 0:
            print("\n Refining prototypes using clustering...")
            bank = build_feature_bank(model, train_loader, device)
            centers = refine_prototypes(bank, num_prototypes=30)
            update_prototypes(model, centers, device)
            print("Prototypes updated\n")

        print(f"Ep: {epoch} avg loss: {loss:.4f}")
    torch.save(model.state_dict(), "model.pth")

if __name__=="__main__":
    main()
