

import torch
import torch.nn as nn
import torch.optim as optim


class FaderNetworksDiscriminator (nn.Module):
    def __init__(self, n_attributes):
        """
        @goal: generate the classifier that deals with the AE- architecture
        
        @input:"n_attributes": Number of attributes used in the latent code.
        @input: "z" : the latent representation 
        @output:"y":the attributes deduced from the "z" latent code 
        """

        super(FaderNetworksDiscriminator, self).__init__()
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
        """
        """
        z= self.dropout(self.relu(self.bn1(self.conv1(z))))        # convolution -> Batch Normalization -> ReLU 
        z= z.view(-1,512)                                          # reshaping to making the layer flat for the next layer 
        z= self.dropout(self.relu(self.layer1(z)))                 # propagation throw the first layer fully connected
        z=self.layer2(z)                                           # propagation throw the second layer fully connected
        y_hat=self.sigmoid(z)                                      # finding the attributes throw the discriminator 

        return y_hat
    


    def step(self, dataloader, num_epochs=5, learning_rate=0.001):
        # Set the discriminator in training mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()

            for batch in dataloader:
                z, attributes = batch
                z, attributes = z.to(device), attributes.to(device)

                optimizer.zero_grad()
                outputs = self(z)
                loss = criterion(outputs, attributes.float())
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
