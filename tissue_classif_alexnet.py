# -*- coding: utf-8 -*-

pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from google.colab import drive
import torch.optim as optim
import os
import copy
from torch.optim import lr_scheduler
from torchsampler import ImbalancedDatasetSampler
drive.mount("/content/gdrive")

device = torch.device("cuda:0" if torch.cuda.is_available() else"cpu")

model_conv = models.alexnet(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False

model_conv.classifier[6] = nn.Linear(in_features=4096, out_features=3, bias=True)
model_conv=model_conv.to(device)

criterion=nn.CrossEntropyLoss()

optimizer_conv=optim.SGD(params=model_conv.parameters(),lr=0.01)
exp_lr_scheduler=lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

data_dir = '/content/gdrive/MyDrive/torch_data/'
# Data augmentation and normalization for training


data_transforms = {
 'train': transforms.Compose([
 transforms.Resize(256),                             
 transforms.RandomResizedCrop(227),
 transforms.RandomHorizontalFlip(), 
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
 ]),
 'valid': transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(227),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
 ]),
 'test': transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(227),
 transforms.ToTensor(), 
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
 ])
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
 data_transforms[x]) for x in ['train', 'valid', 'test']}

def train_model(image_datasets,model,criterion,optimizer,scheduler,num_epochs):

    train_loader=torch.utils.data.DataLoader(image_datasets['train'],sampler=ImbalancedDatasetSampler(image_datasets["train"]),batch_size=4,num_workers=4)
    valid_loader=torch.utils.data.DataLoader(image_datasets['valid'],batch_size=4,shuffle=True,num_workers=4)
  
    best_model_wts = copy.deepcopy(model.state_dict())
    best_no_corrects= 0

    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_loader: 
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs,1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        model.eval()
        no_corrects = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                no_corrects+= torch.sum(preds == labels.data)
                
        if no_corrects> best_no_corrects:
            best_no_corrects= no_corrects
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()
        
    model.load_state_dict(best_model_wts)
    return model

trained_model=train_model(image_datasets,model_conv,criterion,optimizer_conv,exp_lr_scheduler,10)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=4,shuffle=True, num_workers=4) for x in ['train', 'valid','test']}
class_number = 3

confusion_matrix1 = torch.zeros(class_number,class_number)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['train']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = trained_model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix1[t.long(),p.long()] += 1

print(confusion_matrix1)
print(confusion_matrix1.diag()/confusion_matrix1.sum(1))

confusion_matrix2 = torch.zeros(class_number,class_number)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['valid']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = trained_model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix2[t.long(),p.long()] += 1

print(confusion_matrix2)
print(confusion_matrix2.diag()/confusion_matrix2.sum(1))

confusion_matrix3 = torch.zeros(class_number,class_number)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['test']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = trained_model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix3[t.long(),p.long()] += 1

print(confusion_matrix3)
print(confusion_matrix3.diag()/confusion_matrix3.sum(1))