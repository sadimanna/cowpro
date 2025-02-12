dataset = GetDatasetSeg(train_files,batch_size=1)


# In[8]:


sf,sm,qf,qm=dataset[0]
# # qm = get_edge(qm)
# im = transforms.ToPILImage()(qf[3].squeeze(0))
# im1 = transforms.ToPILImage()(qm[3].squeeze(0))
# plt.imshow(np.array(im)*np.array(im1),cmap="gray")
# # plt.imshow(im1,cmap="gray")
# # img = read_img_npy(train_files[0][0])
plt.imshow(sf.squeeze().cpu().numpy(),cmap='gray',vmin = 0.0, vmax = 1.0)
plt.show()
plt.imshow(qf.squeeze().cpu().numpy(),cmap='gray',vmin = 0.0, vmax = 1.0)
plt.show()
