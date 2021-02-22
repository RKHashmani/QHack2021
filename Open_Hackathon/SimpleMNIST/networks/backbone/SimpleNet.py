import torch.nn as nn

class SimpleNet(nn.Module):
  def __init__(self):
    super(SimpleNet, self).__init__()
    # <<< YOUR CODE HERE >>> #
    
    ###################################################################################################################
    # Author: Raheem Hashmani                                                                                         #
    #                                                                                                                 #
    # A simple custom backbone for the MNIST dataset. This architecture was inspired by the one demonstrated in the   #
    # METU CENG 783 course.                                                                                           #
    #                                                                                                                 #
    # The key difference is the use of nn.AdaptiveMaxPool2d, which takes in any arbitrary sized array and outputs an  #
    # array of the desired output size. It does so by automatically defining the correct kernel size and stride       #
    # length. This can be done by first formulating equations that reflect the results of pooling:                    #
    #                                                                                                                 #
    # W_out = (W_in - K)*S + 1 and H_out = (H_in - K)*S + 1                                                           #
    #                                                                                                                 #
    # where W_out and H_out are output size, W_in and H_in are input size, K is the kernel size (1 dimensional), and  #
    # S is the stride length (1 dimensional). Rearranging these, we get:                                              #
    #                                                                                                                 #
    # K = W_in - S*(W_out - 1) and S = (H_in - W_in)/(H_out - W_out)                                                  #
    #                                                                                                                 #
    # For non-square input and output shapes, these work fine, but if any of the shapes are squares, the latter       #
    # equation fails. However, in this case, we can use:                                                              #
    #                                                                                                                 #
    # S = floor[H_in / H_out] (and substitute into K's equation, as normal)                                           #
    #                                                                                                                 #
    # This is possible because, if H_in and H_out are integer multiples, the kernel neatly fits without any overlaps, #
    # and if they are not, there are some overlaps in the kernel, but it does not affect the overall results too much.#
    #                                                                                                                 #
    # This adaptive pooling is used so that, regardless of the input shape (and by extension, the final convolution's #
    # output shape), a kernel size and stride length can be calculated such that the input size to the first fully    #
    # connected layer is constant. Thus, this will allow our ConvNet to work with arbitrary input image sizes.        #
    #                                                                                                                 #
    ###################################################################################################################
    
    
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
    self.pool = nn.MaxPool2d(2)
    self.AdaptPool = nn.AdaptiveMaxPool2d(4)
    self.fc1 = nn.Linear(640, 64)
    self.fc2 = nn.Linear(64, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    
    x = self.pool(self.relu(self.conv1(x)))        
    x = self.pool(self.relu(self.conv2(x)))
    x = self.AdaptPool(self.relu(self.conv3(x))) # The output shape will always be 40*4*4 = 640.
    x = x.view(-1, 640)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x
