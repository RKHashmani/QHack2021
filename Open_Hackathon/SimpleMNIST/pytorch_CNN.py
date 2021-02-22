import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Import the network/backbone to be used
#from networks.backbone.SimpleNet import SimpleNet
from networks.backbone.QNet import SimpleNet

# Define the "device". If GPU is available, device is set to use it, otherwise CPU will be used. 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# To randomly transform the image.
rand_transform = transforms.Compose([transforms.RandomChoice([
      transforms.Pad(3),
      transforms.RandomCrop(26),
      transforms.Pad(1),
      transforms.RandomCrop(27), 
      ]), transforms.ToTensor()])

# To download and setup the train/test dataset
train_data = datasets.MNIST(root = './data', train = True,
    transform = rand_transform, download = True)

test_data = datasets.MNIST(root = './data', train = False,
    transform = rand_transform, download = True)


batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset =  test_data, batch_size = batch_size, shuffle = False)

# Loading the model
net = SimpleNet().to(device)
print(net)

# Preparation for training
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( net.parameters(), lr=1.e-3)

# Training
num_epochs = 2
num_iters_per_epoch = 10 # use only 5K iterations

for epoch in range(num_epochs):
  for i ,(images,labels) in enumerate(train_loader):
    if i==num_iters_per_epoch:
        break

    images = images.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    output = net(images)
    loss = loss_fun(output, labels)
    loss.backward()
    optimizer.step()
    
    if (i+1) % (num_iters_per_epoch // 10) == 0:
      print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, num_iters_per_epoch, loss.item()))


# Testing
correct = 0
total = 0
for images,labels in test_loader:
  images = images.to(device)
  labels = labels.to(device)
  
  out = net(images)
  _, predicted_labels = torch.max(out,1)
  correct += (predicted_labels == labels).sum()
  total += labels.size(0)

print('Percent correct: %.3f'%((100.0*correct)/(total+1)))
