

import torchvision.transforms.functional as ttf

class GetDataset(Dataset):
  def __init__(self,index_list,batch_size,transform1=None,transform2=None):
    self.transform1 = transform1
    self.transform2 = transform2
    self.index_list = index_list
    self.batch_size=batch_size

  
  def __len__(self):
    return len(self.index_list)
  
  def __getitem__(self,idx):
    path = random.choice(self.index_list)
    feat = read_img_npy(path[0])
    mask = read_img_npy(path[1])
    #print("Number of Frames :",feat.shape[0])
    #not_zero = path[2]
    t = transforms.Resize((256,256))
    
    not_zero_range =list(range(0,feat.shape[0]-1))
    p = np.random.rand()
    ind = random.choice(not_zero_range)
    
    div_interval = feat.shape[0]//5
    first_ind = div_interval//2
    interval = first_ind
    ind = random.choice(list(range(5)))
    ind = first_ind + ind*div_interval
    
    support_feat = remove_noise(normalize(feat[ind,:,:].astype('float32')))
    support_mask = mask[ind,:,:].astype('float32')
    support_feat,support_mask = pad(support_feat,support_mask)
    support_feat = t(torch.from_numpy(support_feat[np.newaxis,:,:]))
    support_mask = t(torch.from_numpy(support_mask[np.newaxis,:,:]))
    
    
    query_ind = [random.choice(range(max(0,ind-interval),min(ind+interval,feat.shape[0]-1)))] #-3*(self.batch_size-1))))]
    # query_ind = list(range(query_ind,query_ind+3*(self.batch_size-1)+1,3))
    query_feat = [remove_noise(normalize(feat[i,:,:].astype('float32'))) for i in query_ind]
    query_mask = [mask[i,:,:].astype('float32') for i in query_ind]
    
    #print("Support Index: ",ind, "Query Index: ",query_ind[0])
    
    for i in range(len(query_feat)):
        query_feat[i],query_mask[i] = pad(query_feat[i],query_mask[i])
        query_feat[i] = t(torch.from_numpy(query_feat[i][np.newaxis,:,:]))
        query_mask[i] = t(torch.from_numpy(query_mask[i][np.newaxis,:,:]))
        #query_feat[i],query_mask[i]=
        #AffineTransform(query_feat[i].unsqueeze(0),query_mask[i].unsqueeze(0),[-40,40.0],[0.2,0.2],[0.8,1.2])
        #p = random.random()
#         if p<0.5:
#             query_feat[i],query_mask[i]=crop(query_feat[i].squeeze(0).squeeze(0),query_mask[i].squeeze(0).squeeze(0))
#             query_feat[i],query_mask[i]=query_feat[i].unsqueeze(0).unsqueeze(0),query_mask[i].unsqueeze(0).unsqueeze(0)
        #query_feat[i],query_mask[i]=query_feat[i].squeeze(0),query_mask[i].squeeze(0)
    
    support_feat = torch.stack([support_feat for i in range(self.batch_size)],axis=0)
    support_mask = torch.stack([support_mask for i in range(self.batch_size)],axis=0)
    query_feat = torch.stack(query_feat,axis=0)
    query_mask = torch.stack(query_mask,axis=0)
    
    
    support_mask = support_mask.to(device)
    support_feat = support_feat.to(device)
    query_feat = query_feat.to(device)
    query_mask = query_mask.to(device)
    
    # CHANGE THIS to change the organ
    # make a dictionary maybe??
    
    support_mask[support_mask>1.5]=0;
    support_mask[support_mask>0.1]=1
    query_mask[query_mask>1.5]=0;
    query_mask[query_mask>0.1]=1
    
    if self.transform1 is not None:
        support_feat,query_feat = self.transform1(support_feat),self.transform1(query_feat)
    if self.transform2 is not None:
        support_mask,query_mask = self.transform2(support_mask),self.transform2(query_mask)

    return support_feat,support_mask,query_feat,query_mask

class GetDatasetSeg(Dataset):
  def __init__(self,index_list,batch_size,transform1=None,transform2=None):
    self.transform1 = transform1
    self.transform2 = transform2
    self.index_list = index_list
    self.batch_size=batch_size
    self.t2 = transforms.RandomApply([transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.2)], p = 0.5)
    self.t3 = transforms.RandomApply([transforms.GaussianBlur(3)], p =0.3)
  
  def __len__(self):
    return len(self.index_list)
  
  def __getitem__(self,idx):
    path = self.index_list[idx]
#     key = path[0].split('/')[-1].split('_')[0]    query_path = path#random.choice(organ_dict[key])
    query_path = path
#     feat = read_img_npy(path[0])
#     mask = read_img_npy(path[1])
#     not_zero = path[2]
#     t = transforms.RandomChoice([transforms.Resize((384,384)),transforms.Resize((256,256))])
    
    query_feats = read_img_npy(query_path[0])
    query_masks = read_img_npy(query_path[1])
    #query_not_zero = query_path[2]
    
    not_zero_range = list(range(0,query_feats.shape[0]))#list(range(query_not_zero[0],query_not_zero[1]+1))
    # p = np.random.rand()
    # print(query_feats.shape)
    ind = random.choice(not_zero_range)
    support_feat = []
    query_feat = []
    support_mask = []
    query_mask = []
    
    temp_feat = remove_noise(normalize(query_feats[ind])) #512x512
#     print(temp_feat.shape)
#     plt.imshow(temp_feat)
#     plt.show()
    # print(np.mean(temp_feat), np.min(temp_feat), np.max(temp_feat))
    
    temp=return_segmentation(remove_noise(query_feats[ind]),100,20).astype('float32')
#     print(np.min(temp_feat),np.max(temp_feat))
#     print(np.min(temp),np.max(temp))
#     plt.imshow(temp, cmap='gray',vmin = 0.0, vmax = 1.0)
#     plt.show()
    cnt =0
#     print("seg area",np.sum(temp),np.sum(temp*temp_feat))
#     print(np.sum(temp*temp_feat)/np.sum(temp),(np.sum(temp*temp_feat)/np.sum(temp))<0.02)
#     print(np.sum(temp)/(512**2),(np.sum(temp)/(512**2))<0.05)
#     print(np.sum(temp)/(512**2),(np.sum(temp)/(512**2))>0.9)
    while ((np.sum(temp*temp_feat)/np.sum(temp))<0.05) or (((np.sum(temp)/(512**2))<0.02) or ((np.sum(temp)/(512**2))>0.9)) and cnt<10:
        cnt+=1
        temp=return_segmentation(remove_noise(query_feats[ind]),100,20).astype('float32')
#         print(np.min(temp),np.max(temp))
#         print(np.sum(temp),np.sum(temp*temp_feat))
#         print(np.sum(temp*temp_feat)/np.sum(temp),(np.sum(temp*temp_feat)/np.sum(temp))<0.02)
#         print(np.sum(temp)/(512**2),(np.sum(temp)/(512**2))<0.05)
#         print(np.sum(temp)/(512**2),(np.sum(temp)/(512**2))>0.9)
#     plt.imshow(temp_feat, cmap='gray',vmin = 0.0, vmax = 1.0)
#     plt.show()
#     print(temp_feat.shape, temp.shape)
#     temp_feat, temp = pad(temp_feat, temp)
#     plt.imshow(temp_feat, cmap='gray',vmin = 0.0, vmax = 1.0)
#     plt.show()
    
#     print(temp_feat.shape, temp.shape)
    t = transforms.Resize((320,320))

    temp_feat, temp = t(torch.from_numpy(temp_feat).unsqueeze(0)), t(torch.from_numpy(temp).unsqueeze(0))
    
    f,m = AffineTransform(temp_feat,temp,[-30,30],[0.0,0.0],[0.8,1.2])
#     print(f.shape)
#     plt.imshow(f.squeeze())
#     plt.show()
    # t = random.choice([transforms.Resize((384,384)),transforms.Resize((256,256))])
#     support_feat.append(t(f).squeeze(0))
#     support_mask.append(t(m).squeeze(0))
    # if min(temp_feat.shape[0],temp_feat.shape[1]) < 320:
    
    support_feat.append(f.squeeze(0))
    support_mask.append(m.squeeze(0))
    #print(len(support_mask), support_mask[0].shape)
    # support_feat[0],support_mask[0] = pad(support_feat[0],support_mask[0])
    #print(len(support_mask), support_mask[0].shape)
    
#     print(len(support_mask), support_mask[0].shape)
#     plt.imshow(support_feat[0])
#     plt.show()
    
    sH = 320 #support_feat[0].shape[0]
    sW = 320 #support_feat[0].shape[1]
    crop_param_top = int((sH-256)*random.random())
    crop_param_left = int((sW-256)*random.random())
    
    support_feat[0] = torch.as_tensor(support_feat[0]).unsqueeze(0).unsqueeze(0)
    support_feat[0] = self.t3(self.t2(ttf.crop(support_feat[0], crop_param_top, crop_param_left, 256, 256))).squeeze(0).squeeze(0);
    # print(len(support_feat), support_feat[0].shape)
    
    support_mask[0] = torch.as_tensor(support_mask[0]).unsqueeze(0).unsqueeze(0)
    support_mask[0] = ttf.crop(support_mask[0], crop_param_top, crop_param_left, 256, 256).squeeze(0).squeeze(0);
    
#     plt.imshow(support_feat[0],cmap='gray')
#     plt.show()
#     plt.imshow(support_mask[0],cmap='gray')
#     plt.show()
    
    f,m=AffineTransform(temp_feat,temp,[-30,30],[0.0,0.0],[0.7,1.2])#self.crop(support_feat[0],support_mask[0])
    f,m=f.unsqueeze(0),m.unsqueeze(0)
    
    # t = random.choice([transforms.Resize((384,384)),transforms.Resize((256,256))])
    # f,m = t(f),t(m)
    
    sH = 320 #f.shape[-2]
    sW = 320 #f.shape[-1]
    crop_param_top = int((sH-256)*random.random())
    crop_param_left = int((sW-256)*random.random())
    
    f = self.t3(self.t2(ttf.crop(f, crop_param_top, crop_param_left, 256, 256))).squeeze(0).squeeze(0);
    m = ttf.crop(m, crop_param_top, crop_param_left, 256, 256).squeeze(0).squeeze(0);
    query_feat.append(f)
    query_mask.append(m)
    
#     plt.imshow(query_feat[0],cmap='gray')
#     plt.show()
#     plt.imshow(query_mask[0],cmap='gray')
#     plt.show()
    
    #print(f.shape)
    
#     for i in range(1,self.batch_size):
#         sf,sm = query_feat[i-1],query_mask[i-1]#self.crop(support_feat[i-1],support_mask[i-1])
#         qf,qm = AffineTransform(query_feat[i-1].unsqueeze(0),query_mask[i-1].unsqueeze(0),[-50,50],[0.2,0.2],[0.8,1.2])#self.crop(query_feat[i-1],query_mask[i-1])
#         qf,qm=qf.squeeze(0),qm.squeeze(0)
#         qf = t2(qf)
#         p = random.random()
#         support_feat.append(sf);
#         support_mask.append(sm);
#         query_feat.append(qf);
#         query_mask.append(qm);
        
#     for i in range(self.batch_size):
#         # query_feat[i] = t2(query_feat[i])
#         p = random.random()
#         if p<0.4:
#             query_feat[i],query_mask[i]=crop(query_feat[i],query_mask[i])
#     for i in range(0,self.batch_size):
#         ind = random.choice(not_zero_range)
#         support_feat.append(remove_noise(normalize(query_feats[ind])))
#         temp=return_segmentation(remove_noise(query_feats[ind]),100,50).astype('float32')
# #         while np.sum(temp*(support_feat[i]>0))<0.02*256*256:
# #             temp=return_segmentation(remove_noise(query_feats[ind]),100,50).astype('float32')
#         support_mask.append(temp)
#         support_feat[i],support_mask[i] = pad(support_feat[i],support_mask[i])
#         support_feat[i]=t(torch.as_tensor(support_feat[i]).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0);
#         support_mask[i]=t(torch.as_tensor(support_mask[i]).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0);
        
#         f,m=AffineTransform(support_feat[i].unsqueeze(0),support_mask[i].unsqueeze(0),[-50,50],[0.2,0.2],[0.8,1.2])#self.crop(support_feat[0],support_mask[0])
#         f,m=f.squeeze(0),m.squeeze(0)
#         p = random.random()
#         if p<0.3:
#             f,m=crop(f,m)
#         query_feat.append(f)
#         query_mask.append(m)
        #print(query_feat[i].shape)
#         if p>0.4 and p<0.7:
#             resize_to = int(max(0.8,random.random())*256)
#             query_feat[i]=ttf.pad(ttf.resize(query_feat[i],resize_to),256-resize_to)
#             query_mask[i]=ttf.pad(ttf.resize(query_mask[i],resize_to),256-resize_to)
    
    
    query_feat = torch.stack(query_feat,axis=0)
    query_mask = torch.stack(query_mask,axis=0)
    support_feat = torch.stack(support_feat,axis=0)
    support_mask = torch.stack(support_mask,axis=0)
    
    query_feat = query_feat.unsqueeze(1)
    support_feat = support_feat.unsqueeze(1)
    query_mask = query_mask.unsqueeze(1)
    support_mask = support_mask.unsqueeze(1)
    

    support_mask = support_mask.to(device).float()
    support_feat = support_feat.to(device).float()
    query_feat = query_feat.to(device).float()
    query_mask = query_mask.to(device).float()
    

#     support_mask[support_mask>1.5]=0;support_mask[support_mask>0.1]=1
#     query_mask[query_mask>1.5]=0;query_mask[query_mask>0.1]=1
    
#     if self.transform1 is not None:
#         support_feat,query_feat = self.transform1(support_feat),self.transform1(query_feat)
#     if self.transform2 is not None:
#         support_mask,query_mask = self.transform2(support_mask),self.transform2(query_mask)

    return support_feat,support_mask,query_feat,query_mask