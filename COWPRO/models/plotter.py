def plot_(support_feat,support_mask,query_feat,query_mask, pred_mask):
    support_feat = support_feat.cpu().squeeze()
    support_mask = support_mask.cpu().squeeze()
    query_feat = query_feat.cpu().squeeze()
    query_mask = query_mask.cpu().squeeze()

    # model.train()
    pred_mask = pred_mask.detach().cpu().squeeze()
    d = dice_loss(pred_mask,query_mask)
    print("Dice Score with Query: ",(torch.sum(2*pred_mask*query_mask)+1)/(torch.sum(pred_mask)+torch.sum(query_mask)+1))
    print("Dice Score with Support: ",(torch.sum(2*pred_mask*support_mask)+1)/(torch.sum(pred_mask)+torch.sum(support_mask)+1))
    print(1-d)

    pred_mask = 0.65*(pred_mask > 0.5)
    # pred_mask[pred_mask==0] = 0.35
    pred_mask = torch.stack([pred_mask,0.0*torch.ones(pred_mask.shape),0.0*torch.ones(pred_mask.shape)],dim = 0)
    pred_mask = pred_mask.to(dtype=torch.float32)
    # 
    # pred_mask = pred_mask*query_feat
    # print(support_mask.shape)

    support_mask = torch.stack([torch.zeros(support_mask.shape),torch.zeros(support_mask.shape),support_mask*0.65], dim = 0)
    query_mask = torch.stack([torch.zeros(query_mask.shape),query_mask*0.65,torch.zeros(query_mask.shape)], dim = 0)

    support_img = transforms.ToPILImage()(support_feat)
    support_mask_img = transforms.ToPILImage()(support_mask)
    query_feat_img = transforms.ToPILImage()(query_feat)
    query_mask_img = transforms.ToPILImage()(query_mask)
    pred_mask_img = transforms.ToPILImage()(pred_mask) #.squeeze())

    fig = plt.figure(figsize=(15,10))
    rows = 3
    columns=2

    fig.add_subplot(rows,columns,1)
    plt.imshow(support_img,cmap='gray')
    plt.axis("off")
    plt.title("Support image")

    fig.add_subplot(rows,columns,2)
    plt.imshow(support_img,cmap='gray', alpha = 0.8)
    plt.imshow(support_mask_img, alpha = 0.6)
    plt.axis("off")
    plt.title("Support mask")

    fig.add_subplot(rows,columns,3)
    plt.imshow(query_feat_img,cmap='gray')
    plt.axis("off")
    plt.title("Query image")

    fig.add_subplot(rows,columns,4)
    plt.imshow(query_feat_img,cmap='gray', alpha = 0.8)
    plt.imshow(query_mask_img, alpha = 0.6)
    plt.axis("off")
    plt.title("Query mask")

    fig.add_subplot(rows,columns,5)
    plt.imshow(query_feat_img, cmap = 'gray', alpha = 0.8)
    #plt.imshow(support_mask_img, alpha = 0.8)
    plt.imshow(query_mask_img, alpha = 0.4)
    plt.imshow(pred_mask_img, alpha = 0.4)
    plt.axis("off")
    plt.title("Predicted mask")

    fig.add_subplot(rows,columns,6)
    plt.imshow(query_feat_img, cmap = 'gray', alpha = 0.8)
    plt.imshow(support_mask_img, alpha = 0.4)
    #plt.imshow(query_mask_img, alpha = 0.7)
    plt.imshow(pred_mask_img, alpha = 0.4)
    plt.axis("off")
    plt.title("Predicted mask")
    
    plt.show()