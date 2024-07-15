

class ResNetLayer(pl.LightningModule):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super().__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return


    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch



class ResNet(pl.LightningModule):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 64, stride=2)
        self.layer3 = ResNetLayer(64, 64, stride=2)
        #self.layer4 = ResNetLayer(64, 64, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))
        
        return


    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        #batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class GlobalLayerNorm(pl.LightningModule):
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y

class visualFrontend(pl.LightningModule): # Activated visualTCN & Visualconv1d

    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self, context_dim=64):
        super().__init__()
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                        )
        self.resnet = ResNet(context_dim*2)
        
        self.visualTCN       = visualTCN(context_dim)      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D(context_dim)
        self.context_dim = context_dim
        return


    def forward(self, inputBatch):
        if inputBatch.ndim!=4:
            B=1
            #print('inputbatch shape', inputBatch.shape)
            #T = inputBatch.shape[0]
            T, W, H = inputBatch.shape
            inputBatch = inputBatch.view(T, 1, 1, W, H)

            #inputBatch=np.expand_dims(inputBatch, 1)
            #inputBatch=np.expand_dims(inputBatch, 1)
            
        else:
            B, T, W, H = inputBatch.shape
            inputBatch = inputBatch.view(B*T, 1, 1, W, H)

        inputBatch = (inputBatch / 255 - 0.4161) / 0.1688

        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        #inputBatch = np.transpose(inputBatch, (1,2,0,3,4))

        batchsize = inputBatch.shape[0] #input batch shape: [1,1,600,112,112] 600이 배치 시간print(')
        batch = self.frontend3D(inputBatch) #after [1,64,51,28,28]

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]) # 600,64,28,28
        #import pdb; pdb.set_trace()
        outputBatch = self.resnet(batch)
        
        
        x = outputBatch.view(B, T, self.context_dim*2) #512)         #[8,75,512]
        # Activate below
        
        x = x.transpose(1,2)     #[8,512,75] [8,64,51]
        x = self.visualTCN(x)
        x = self.visualConv1D(x) #[8,256,75]
        x = x.transpose(1,2) # set time dim.
        
        #print('visual shape', x.shape)
        
        return x

class ResNet(pl.LightningModule):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self, context_dim=64):
        super().__init__()
        self.context_dim = context_dim
        if context_dim==64:
            self.layer1 = ResNetLayer(64, 64, stride=1)
            self.layer2 = ResNetLayer(64, 64, stride=2)
            self.layer3 = ResNetLayer(64, 64, stride=2)
            #self.layer4 = ResNetLayer(64, 64, stride=2)
            self.avgpool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))
        elif context_dim==512:
            self.layer1 = ResNetLayer(64, 64, stride=1)
            self.layer2 = ResNetLayer(64, 128, stride=2)
            self.layer3 = ResNetLayer(128, 256, stride=2)
            self.layer4 = ResNetLayer(256, 512, stride=2) # only this layer is modified
            self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        elif context_dim==128:
            self.layer1 = ResNetLayer(64, 64, stride=1)
            self.layer2 = ResNetLayer(64, 64, stride=2)
            self.layer3 = ResNetLayer(64, 128, stride=2)
        return


    def forward(self, inputBatch):
        #import pdb; pdb.set_trace()
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        if self.context_dim==512:
            batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch



class DSConv1d(pl.LightningModule):
    def __init__(self, context_dim=64):
        super().__init__()
        if context_dim!=64:
            self.net = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Conv1d(512, 512, 3, stride=1, padding=1,dilation=1, groups=512, bias=False),
                nn.PReLU(),
                GlobalLayerNorm(512),
                nn.Conv1d(512, 512, 1, bias=False),
                )
        else:
            self.net = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(context_dim*2),
                nn.Conv1d(context_dim*2, context_dim*2, 3, stride=1, padding=1, dilation=1, groups=context_dim*2, bias=False),
                nn.PReLU(),
                GlobalLayerNorm(context_dim*2),
                nn.Conv1d(context_dim*2, context_dim*2, 1, bias=False),
                )

    def forward(self, x):
        out = self.net(x)
        return out + x

class visualTCN(pl.LightningModule):
    def __init__(self, context_dim=None):
        super().__init__()
        stacks = []        
        if context_dim is None:
            for x in range(5):
                stacks += [DSConv1d(None)]
        else:
            for x in range(5):
                stacks += [DSConv1d(context_dim*2)]
        self.net = nn.Sequential(*stacks) # Visual Temporal Network V-TCN

    def forward(self, x):
        out = self.net(x)
        return out

class visualConv1D(pl.LightningModule):
    def __init__(self, context_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(context_dim*2, context_dim, 5, stride=1, padding=2),
            nn.BatchNorm1d(context_dim),
            nn.ReLU(),
            )

    def forward(self, x):
        out = self.net(x)
        return out

