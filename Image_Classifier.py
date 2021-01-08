import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from google.colab import drive

drive.mount('/content/gdrive')

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
train_data = datasets.ImageFolder(root= "/content/gdrive/My Drive/diff_image_stamps",transform=transform)
loader_train = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)

train_data.class_to_idx
classes = ('bogus', 'real')
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      img = image_batch[n] / 2 + 0.5     
      img = img.numpy()
      plt.imshow(np.transpose(img, (1, 2, 0)))
      plt.title(classes[label_batch[n]])
      plt.axis('off')

sample_images, sample_labels = next(iter(loader_train))
show_batch(sample_images, sample_labels)

data_size = len(train_data)
data_idx = list(range(data_size))
np.random.shuffle(data_idx)
split_tt = int(np.floor(0.1 * data_size))  
train_idx, test_idx = data_idx[split_tt:], data_idx[:split_tt]

train_size = len(train_idx)
indices_train = list(range(train_size))
np.random.shuffle(indices_train)
split_tv = int(np.floor(0.2 * train_size)) #Split train and validation
train_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=False, sampler=train_sampler, num_workers=10)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=False, sampler=valid_sampler, num_workers=10)
test_loader = torch.utils.data.DataLoader(train_data,  batch_size=4, shuffle=False, sampler=test_sampler, num_workers=10)

###Mod1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


mod1 = Net()


criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(mod1.parameters(), lr=0.001, momentum=0.9)

###Mod2
mod2 = models.resnet18(pretrained=True)
mod2.fc.out_features = 2
criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.SGD(mod2.fc.parameters(), lr=0.001, momentum=0.9)

###Mod3
mod3 = models.resnet18(pretrained=True)
for param in mod3.parameters():
    param.requires_grad = False
    
mod3.fc = nn.Sequential(nn.Linear(512, 10),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(10, 2),
                                 nn.LogSoftmax(dim=1))

criterion3 = nn.CrossEntropyLoss()
optimizer3 = optim.Adam(mod3.fc.parameters(), lr=0.001)

###Training

def Training(model, training_set, optimizer, criterion):
    model.train()
    train_running_correct=0.0
    total=0.0
    train_running_loss=0.0
    for i, data in enumerate(training_set, 0):
      images, label = data
    
      optimizer.zero_grad()
    
      outputs = model(images)
      loss = criterion(outputs, label)
      loss.backward()
      optimizer.step()
      
      train_running_loss += loss.item()
      _, preds = torch.max(outputs.data, 1)
      train_running_correct+=(preds==label).sum().item()
      
      total+=label.size(0)
      
    train_loss = train_running_loss/total
    train_correct = 100*train_running_correct/total
    print(f'Epoch: {epoch + 1}; Training accuracy: {train_correct}; Loss: {train_loss}')

    return train_loss, train_correct


###Validation

def Validation(model, validation_set, criterion):
    model.eval()
    val_running_correct=0.0
    total=0.0
    val_running_loss=0.0
    for i, data in enumerate(validation_set, 0):
      images, label = data
    
      outputs = model(images)
      loss = criterion(outputs, label)
      
      val_running_loss += loss.item()  
      _, preds = torch.max(outputs.data, 1)
      val_running_correct+=(preds==label).sum().item()
      
      total+=label.size(0)
 
    val_loss = val_running_loss/total
    val_correct = 100*val_running_correct/total
    print(f'Epoch: {epoch + 1}; Validation accuracy: {val_correct}; Loss: {val_loss}')

    return val_loss, val_correct


###Test

def Testing(model, test_dataset):
  dataiter = iter(test_dataset)
  images, labels = dataiter.next()
  correct = 0
  total = 0
  with torch.no_grad():
      for data in test_dataset:
          images, labels = data
          outputs = model(images)  
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
   
  def imshow(img):
      img = img / 2 + 0.5     
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))
      plt.show()

  imshow(torchvision.utils.make_grid(images)) 
  print('Label: ', ' '.join('%5s' % classes[labels[j]] for j in range(3)))
  print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(3)))
  print('Testing accuracy : %d %%' % (100 * correct / total))

##Check:

train_loss , train_correct = [], []
val_loss , val_correct = [], []
start = time.time()
for epoch in range(10):
    train_epoch_loss, train_epoch_correct = Training(mod1, train_loader, optimizer1, criterion1) #To change: model, optimizer and criterion
    val_epoch_loss, val_epoch_correct = Validation(mod1, valid_loader, criterion1) #To change: model and criterion
    train_loss.append(train_epoch_loss)
    train_correct.append(train_epoch_correct)
    val_loss.append(val_epoch_loss)
    val_correct.append(val_epoch_correct)
end = time.time()
print('Finished Process')
print((end-start)/60, 'minutes')

Testing(mod1, test_loader) #To change: model

##Visualize:

#Loss
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.legend()
plt.show()

#Correct:
plt.figure(figsize=(10, 7))
plt.plot(train_correct, color='green', label='train correct')
plt.plot(val_correct, color='blue', label='validataion correct')
plt.legend()
plt.show()
