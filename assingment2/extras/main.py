import torch
import torch.nn as nn
import pickle
import os
from encoder import *
from decoder import *
from dataset import *
from utils import *
from seq2seq import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ARGS():
    def __init__(self):
        self.model_type = ""
        self.data_dir = "./data"
        self.batch_size = 16
        self.num_workers = 4
        self.epochs = 100
        self.en_hidden = 512
        self.de_hidden = 512
        self.en_num_layers = 2
        self.de_num_layers = 2
        self.embed_dim = 300
        self.processed_data = "./processed_data"

if __name__ == "__main__":
    print(device)
    
    args = ARGS()
    pth = os.getcwd()
    args.processed_data = os.path.join(pth, "processed_data")
    args.data_dir = os.path.join(pth, "data")
    build_vocab(args.processed_data)
    # train_dataset = TextToMathDataset(args.processed_data, "train")
    # model = Seq2Seq(args).to(device)

    # criterion = nn.CrossEntropyLoss(ignore_index = 0)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)  #adam or SGD ??
    # schedulers = {
    #     "stepLR" : torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=- 1, verbose=False),
    #     "cosineLR" : torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, verbose=False)
    # }
    # current_scheduler = schedulers["cosineLR"]
    # train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, num_workers = args.num_workers, collate_fn=collate)

    # for epoch in range(args.epochs):
    #     print("Epoch = ", epoch)
    #     model.train()
    #     total_loss = 0
    #     for i, batch in enumerate(train_loader):
    #         optimizer.zero_grad()
    #         question = batch["question"].to(device)
    #         problem = batch["problem"].to(device)
    #         output = model(question, problem)
    #         #print output shape

    #         print("Output Shape = ", output.shape)
    #         loss = criterion(output, problem)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         if i % 100 == 0:
    #             print(f"Epoch {epoch}, Loss = {total_loss/(i+1)}")
    #     current_scheduler.step()
    #     print(f"Epoch {epoch}, Loss = {total_loss/len(train_loader)}")
    #     torch.save(model.state_dict(), f"model_{epoch}.pth")
    #     print("Model Saved")
    # print("Training Complete")