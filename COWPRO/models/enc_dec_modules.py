class encoder(nn.Module):
    def __init__(self,in_channels,base_c = 64, use_sigmoid=None):
        super().__init__()
        self.in_c = in_channels
        self.base_c = base_c

        self.sigmoid = use_sigmoid
        if use_sigmoid is not None:
            self.sigmoid = nn.Sigmoid()

        # 256 -> 128
        self.block1 = encoder_block(in_channels,self.base_c) #64
        # 128 -> 64
        self.block2 = encoder_block(self.base_c,self.base_c*2) #128
        # 64 -> 32
        self.block3 = encoder_block(self.base_c*2, self.base_c*2) #128
        # 32 -> 16
        self.block4 = encoder_block(self.base_c*2,self.base_c*2) #128
        # 16 -> 8
        self.block5 = encoder_block(self.base_c*2,self.base_c*4) #256
        # 8 -> 4
        self.block6 = encoder_block(self.base_c*4,self.base_c*4) #256
        # self.block6 = encoder_block(128,256)
        # self.block7 = encoder_block(256,512)
        self.avgpool = nn.AdativeAvgPool2d(1) #kernel_size = (2,2), stride = 2)



    def forward(self,x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        out6 = self.block6(out5)
        # out7 = self.block7(out6)

        # if self.sigmoid is not None:
        #     out1 = self.sigmoid(out1)
        #     out2 = self.sigmoid(out2)
        #     out3 = self.sigmoid(out3)
        #     out4 = self.sigmoid(out4)
        #     out5 = self.sigmoid(out5)
        #     out6 = self.sigmoid(out6)
        #     out7 = self.sigmoid(out7)
        return out1,out2,out3,out4,out5,out6, self.avgpool(out6) #,out7

class decoder(nn.Module):
    def __init__(self,in_channels,encoder_feat1, encoder_feat2,encoder_mask,base_c = None,sigmoid=None):
        super().__init__()
        self.feat1 = encoder_feat1
        self.feat2 = encoder_feat2
        self.mask = encoder_mask
        self.base_c = base_c if base_c is not None else in_channels
        self.t1 = conv_transpose_block(in_channels,self.base_c,True) #256
        self.t2 = conv_transpose_block(self.base_c*2,self.base_c,True) #512 - 256
        self.t3 = conv_transpose_block(3*self.base_c//2,self.base_c//2,True) #512 - 128
        self.t4 = conv_transpose_block(self.base_c,self.base_c//2,True) #256 - 128
        self.t5 = conv_transpose_block(self.base_c,self.base_c//2,True) #256 - 128
        self.t6 = conv_transpose_block(3*self.base_c//4,self.base_c//2,True) #192 - 128
        # self.t0 = conv_transpose_block(56,28,True)
        
        # self.u6 =QKVAttention(512)
        # self.u5 = QKVAttention(self.base_c)
        self.u4 =  QKVAttention(self.base_c//2) #LBAttention(self.base_c, 32*2)
        self.u3 =  QKVAttention(self.base_c//2) #LBAttention(self.base_c//2, 64**2)
        self.u2 =  QKVAttention(self.base_c//2) #LBAttention(self.base_c//2, 128**2)
        self.u1 =  QKVAttention(self.base_c//4) #LBAttention(self.base_c//4, 256**2)
        
        self.final = nn.Conv2d(self.base_c//2,1,kernel_size=1)
        
        self.sigmoid = nn.Sigmoid() if sigmoid is not None else None
        self.flatten = nn.Flatten()
    
    
    def forward(self,support_feat,support_mask,query_feat):
        sf1,sf2,sf3,sf4,sf5,sf6 = self.feat1(support_feat)
        sm1,sm2,sm3,sm4,sm5,sm6 = self.mask(support_mask)
        qf1,qf2,qf3,qf4,qf5,qf6 = self.feat2(query_feat)

        cat1 = qf1 #self.u1(sf1,sm1,sf1)
        cat2 = self.u2(sf2, sm2, sf2)
        cat3 = self.u3(sf3,sm3,sf3)
        cat4 = self.u4(sf4,sm4,sf4)
        cat5 = qf5 # self.u5(sf5,sm5,sf5)
        cat6 = qf6 # self.u6(sf6,sm6,sf6)
        # cat0 = qf0

        out = self.t1(cat6)
        out = torch.cat([out,cat5],axis=1)

        out = self.t2(out)
        out = torch.cat([out,cat4],axis=1)

        out = self.t3(out)
        out = torch.cat([out,cat3],axis=1)

        out = self.t4(out)
        out = torch.cat([out,cat2],axis=1)

        out = self.t5(out)
        out = torch.cat([out,cat1],axis=1)

        out = self.t6(out)
        # out = torch.cat([out,cat0],axis=1)
        
        # out = self.t0(out)
        #out = self.final_3(out)
        
        mask = self.final(out)
        
        if self.sigmoid is not None:
            mask = self.sigmoid(mask)

        return mask


class Decoder(nn.Module):
    def __init__(self,in_channels,encoder_feat1, encoder_mask,base_c = None,sigmoid=None, temperature = 1.0):
        super().__init__()
        self.feat1 = encoder_feat1
        #self.feat2 = encoder_feat2
        self.mask = encoder_mask
        self.base_c = base_c if base_c is not None else in_channels
        self.t6 = conv_transpose_block(in_channels,self.base_c,False) #512 - 512
        self.t5 = conv_transpose_block(self.base_c,self.base_c//2,False) #512 - 256
        self.t4 = conv_transpose_block(self.base_c//2,self.base_c//2,False) #256 - 256
        self.t3 = conv_transpose_block(self.base_c//2,self.base_c//2,False) #256 - 256
        self.t2 = conv_transpose_block(self.base_c//2,self.base_c//4,False) #256 - 128
        self.t1 = conv_transpose_block(self.base_c//4,self.base_c//4,False) #128 - 128
        # self.t0 = conv_transpose_block(56,28,True)

        # self.u6 = QKVAttention(self.base_c)
        self.u5 = QKVAttention(self.base_c) #512
        self.u4 =  QKVAttention(self.base_c//2) #256
        self.u3 =  QKVAttention(self.base_c//2) #256
        self.u2 =  QKVAttention(self.base_c//2) #256
        self.u1 =  QKVAttention(self.base_c//4) #128

        # self.v6 = QKVAttention(self.base_c)
        self.v5 = QKVAttention(self.base_c) #512
        self.v4 =  QKVAttention(self.base_c//2) #256
        self.v3 =  QKVAttention(self.base_c//2) #256
        self.v2 =  QKVAttention(self.base_c//2) #256
        self.v1 =  QKVAttention(self.base_c//4) #128

        self.w6 = QKVAttention(self.base_c)
        self.w5 = QKVAttention(self.base_c) #512
        self.w4 =  QKVAttention(self.base_c//2) #256
        self.w3 =  QKVAttention(self.base_c//2) #256
        self.w2 =  QKVAttention(self.base_c//2) #256
        self.w1 =  QKVAttention(self.base_c//4) #128

        self.final1 = nn.Conv2d(self.base_c//4,self.base_c//4,kernel_size=3, padding = 'same')
        self.final2 = nn.Conv2d(self.base_c//4,1,kernel_size=3, padding = 'same')

        self.sigmoid = nn.Sigmoid() if sigmoid is not None else None
        self.flatten = nn.Flatten()

        self.temperature = temperature

        self.relu = F.relu

    def calc_cs(self, out, cat):
        out_ = out.mean(dim = (2,3), keepdim = False)
        cat_ = cat.mean(dim = (2,3), keepdim = False)
        out_ = torch.nn.functional.normalize(out_, dim = -1, p = 2)
        cat_ = torch.nn.functional.normalize(cat_, dim = -1, p = 2)
        cs = torch.sum(torch.multiply(out_, cat_))
        return cs


    def forward(self,support_feat,support_mask,query_feat):
        sf1,sf2,sf3,sf4,sf5,sf6 = self.feat1(support_feat)
        sm1,sm2,sm3,sm4,sm5,sm6 = self.mask(support_mask)
        qf1,qf2,qf3,qf4,qf5,qf6 = self.feat1(query_feat)
        # 128, 256, 256, 256, 512, 512
        
        # fig, axs = plt.subplots(1,6)
        # axs[0,0].imshow(sf1.detach().cpu().numpy().mean((0,1)))
        # axs[0,1].imshow(sf2.detach().cpu().numpy().mean((0,1)))
        # axs[0,2].imshow(sf3.detach().cpu().numpy().mean((0,1)))
        # axs[0,3].imshow(sf4.detach().cpu().numpy().mean((0,1)))
        # axs[0,4].imshow(sf5.detach().cpu().numpy().mean((0,1)))
        # axs[0,5].imshow(sf6.detach().cpu().numpy().mean((0,1)))
        # axs[1,0].imshow(sm1.detach().cpu().numpy().mean((0,1)))
        # axs[1,1].imshow(sm2.detach().cpu().numpy().mean((0,1)))
        # axs[1,2].imshow(sm3.detach().cpu().numpy().mean((0,1)))
        # axs[1,3].imshow(sm4.detach().cpu().numpy().mean((0,1)))
        # axs[1,4].imshow(sm5.detach().cpu().numpy().mean((0,1)))
        # axs[1,5].imshow(sm6.detach().cpu().numpy().mean((0,1)))
        # axs[2,0].imshow(qf1.detach().cpu().numpy().mean((0,1)))
        # axs[2,1].imshow(qf2.detach().cpu().numpy().mean((0,1)))
        # axs[2,2].imshow(qf3.detach().cpu().numpy().mean((0,1)))
        # axs[2,3].imshow(qf4.detach().cpu().numpy().mean((0,1)))
        # axs[2,4].imshow(qf5.detach().cpu().numpy().mean((0,1)))
        # axs[2,5].imshow(qf6.detach().cpu().numpy().mean((0,1)))
        

        cat1 = self.u1(sf1,sm1,sf1) #128
        cat2 = self.u2(sf2,sm2,sf2) #256
        cat3 = self.u3(sf3,sm3,sf3) #256
        cat4 = self.u4(sf4,sm4,sf4) #256
        cat5 = self.u5(sf5,sm5,sf5) #512
        cat6 = self.u6(sf6,sm6,sf6) #512
        # cat0 = qf0
        
        # qf6 # 512
        out6 = self.w6(qf6, cat6, qf6) #512
        
        #decoder output
        out5 = self.t6(self.relu(out6)) #512
        # axs[5].imshow(out5.detach().cpu().numpy().mean((0,1)))
        # query image feature map and query decoder feature map  attention
        out5 = self.v5(qf5, out5, qf5) #512
        
        # cs5 = -self.calc_cs(out5, cat5)
        # Concatenate support vkv and query vkv 
        out5 = self.w5(out5, cat5 ,out5) # torch.cat([out5,cat5],axis=1) #512
        

        out4 = self.t5(self.relu(out5)) #256
        # axs[4].imshow(out4.detach().cpu().numpy().mean((0,1)))
        out4 = self.v4(qf4, out4, qf4) #256
        
        # cs4 = -self.calc_cs(out4, cat4)
        out4 = self.w4(out4, cat4, out4) #torch.cat([out4,cat4],axis=1) #256

        out3 = self.t4(self.relu(out4)) #256
        # axs[3].imshow(out3.detach().cpu().numpy().mean((0,1)))
        out3 = self.v3(qf3, out3, qf3) #256
        
        # cs3 = -self.calc_cs(out3, cat3)
        out3 = self.w3(out3, cat3, out3) #torch.cat([out3,cat3],axis=1) #256

        out2 = self.t3(self.relu(out3)) #256
        # axs[2].imshow(out2.detach().cpu().numpy().mean((0,1)))
        out2 = self.v2(qf2, out2, qf2) #256
        
        # cs2 = -self.calc_cs(out2, cat2)
        out2 = self.w2(out2, cat2, out2) #torch.cat([out2,cat2],axis=1) #256

        out1 = self.t2(self.relu(out2)) #128
        # axs[1].imshow(out1.detach().cpu().numpy().mean((0,1)))
        out1 = self.v1(qf1, out1, qf1) #128
        
        # cs1 = -self.calc_cs(out1, cat1)
        out1 = self.w1(out1, cat1, out1) #torch.cat([out1,cat1],axis=1) #128

        out = self.t1(self.relu(out1)) #128
        # axs[0].imshow(out.detach().cpu().numpy().mean((0,1)))
        # out = torch.cat([out,cat0],axis=1)
        #plt.show()
        # out = self.t0(out)
        #out = self.final_3(out)
        
        mask = self.final2(self.relu(self.final1(out))) #128
        
        if self.sigmoid is not None:
            mask = self.sigmoid(mask/self.temperature)
            
            # plt.imshow(mask.detach().cpu().squeeze().numpy(),cmap='gray')
            # plt.show()
        # print(cs1, cs2, cs3, cs4, cs5)

        return mask, 0 #cs1+cs2+cs3+cs4+cs5
