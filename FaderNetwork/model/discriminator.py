import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_attributes):
        """
        @goal: generate the classifier that deals with the AE- architecture
        
        @input:"n_attributes": Number of attributes used in the latent code.
        @input: "z" : the latent representation 
        @output:"y":the attributes deduced from the "z" latent code 
        """

        super(Discriminator, self).__init__()
        ##Part1:"C_k" layer defined in the paper for the discriminator :  convolution -> Batch Normalization -> ReLU  ,k represents the number of filters(in our case 512)
        self.conv1=nn.Conv2d(512,512,kernel_size=(4,4),stride=(2,2),padding=(1,1)) #values for kernel_size ,stride and padding are defined in the paper
        self.bn1=nn.BatchNorm2d(512) 
        self.relu=nn.ReLU()
        
        ##Part2: the two layers of size of  512  and n_attributes
        self.layer1=nn.Linear(512,512)
        self.layer2=nn.Linear(512,n_attributes)

        self.dropout=nn.Dropout(p=0.3) #value following the paper 
        self.sigmoid=nn.Sigmoid()

    def forward(self, z):
        z= self.dropout(self.relu(self.bn1(self.conv1(z))))        # convolution -> Batch Normalization -> ReLU 
        z= z.view(-1,512)                                          # reshaping to making the layer flat for the next layer 
        z= self.dropout(self.relu(self.layer1(z)))                 # propagation throw the first layer fully connected
        z=self.layer2(z)                                           # propagation throw the second layer fully connected
        y_hat=self.sigmoid(z)                                      # finding the attributes throw the discriminator 

        return y_hat