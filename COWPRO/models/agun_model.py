import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from pdb import set_trace

from .attention_modules import *

from kornia.filters import sobel

# from torchgeometry.losses.tversky import TverskyLoss

class WeightedBCELoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, y, pos_wt, neg_wt):
		pos_loss = - torch.multiply(y, torch.log(x))
		pos_loss = pos_wt * pos_loss
		# pos_loss = pos_loss.mean()
		# print(pos_loss)

		neg_loss = - torch.multiply((1 - y), torch.log(1 - x))
		neg_loss = neg_wt * neg_loss
		# print(neg_loss)

		loss = pos_loss + neg_loss
		loss = loss.mean()

		return loss

class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coeficient loss.

    According to [1], we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \ G| + \beta |G \ P|}

    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Notes:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        # input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        # target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = (input * target).sum() #torch.sum(input * target, dims)
        fps = (input * (1. - target)).sum()
        fns = ((1. - input) * target).sum() 

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)
        return torch.mean(torch.pow(1. - tversky_loss, self.gamma))

class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        probs = logits #torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss

class conv_transpose_block(nn.Module):
    def __init__(self,in_channels,out_channels,use_relu = None):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.use_relu = use_relu

    def forward(self,x):
        x = self.convT(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv(x)
        x = self.bn2(x)

        if self.use_relu is not None:
            x = self.relu2(x)
        return x

class Encoder(nn.Module):
	def __init__(self, use_pretrained = False):

		super().__init__()

		self.use_pretrained = use_pretrained

		self.encoder = torchvision.models.resnet50(pretrained = True) # self.use_pretrained) #weights = 'IMAGENET1K_V2' if use_pretrained else None)
		self.encoder.fc = nn.Identity()

		self.fhooks = []

		self.output_layers = ['layer1', 'layer2', 'layer3', 'layer4']

		self.selected_out = {}

		for i,l in enumerate(list(self.encoder._modules.keys())):
			# print(i,l)
			if l in self.output_layers:
				self.fhooks.append(getattr(self.encoder,l).register_forward_hook(self.forward_hook(l)))
    
	def forward_hook(self,layer_name):
		def hook(module, input, output):
			self.selected_out[layer_name] = output
		return hook

	def forward(self, x):
		x = self.encoder(x)
		# print(self.selected_out.keys())
		return (self.selected_out['layer1'], 
				self.selected_out['layer2'], 
				self.selected_out['layer3'], 
				self.selected_out['layer4'],
				x)

class AGUNet(nn.Module):
	def __init__(self,
				 dec_base_c = 2048,
				 param_out = 12,
				 cfg = None,
				 is_val = False):
		super().__init__()
		self.dec_base_c = dec_base_c
		self.param_out = param_out
		self.image_encoder = self.get_encoder(use_pretained = True)
		self.mask_encoder = self.get_encoder(use_pretained = True)
		self.config = cfg
		self.is_val = is_val
		# self.image_decoder = self.get_decoder(self.dec_base_c)

		# self.regressor = self.get_regressor(self.dec_base_c, self.param_out, bn = True)

		self.l1_loss = nn.L1Loss() #None

		self.bce_loss = nn.BCELoss(reduction = 'none') #WeightedBCELoss() #nn.BCELoss() #None

		self.relu = nn.functional.relu

		self.l2_loss = nn.functional.mse_loss

		self.dice_loss = SoftDiceLossV1()

		self.tversky_loss = TverskyLoss(alpha = torch.tensor(self.config['tversky_alpha'], requires_grad = False, device = 'cuda:0'), 
										beta = torch.tensor(self.config['tversky_beta'], requires_grad = False, device = 'cuda:0'), 
										gamma = torch.tensor(1.0, requires_grad = False, device = 'cuda:0'))

		####################################################
		self.preconv4 = nn.Conv2d(self.dec_base_c, self.dec_base_c//2, kernel_size = 1) #2048 -> 1024
		self.el4_dec = conv_transpose_block(self.dec_base_c//2,self.dec_base_c//2,True) #1024 -> 1024

		self.preconv3 = nn.Conv2d(self.dec_base_c, self.dec_base_c//4, kernel_size = 1) #2048 -> 512
		self.el3_dec = conv_transpose_block(self.dec_base_c//4,self.dec_base_c//4,True) #512 -> 512

		self.preconv2 = nn.Conv2d(self.dec_base_c//2, self.dec_base_c//8, kernel_size = 1) #1024 -> 256
		self.el2_dec = conv_transpose_block(self.dec_base_c//8,self.dec_base_c//8,True) #256 -> 256

		self.preconv1 = nn.Conv2d(self.dec_base_c//4, self.dec_base_c//16, kernel_size = 1) #512 -> 128
		self.el1_dec = conv_transpose_block(self.dec_base_c//16,self.dec_base_c//16,True) #128 -> 128

		self.preconv0 = nn.Conv2d(self.dec_base_c//16, self.dec_base_c//32, kernel_size = 1) #128 -> 64
		self.el0_dec = conv_transpose_block(self.dec_base_c//32, self.dec_base_c//32, True) #64 -> 64
		self.decfin_out = nn.Conv2d(self.dec_base_c//32,1,kernel_size=1) #64 -> 1
		###################################################
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim = 1) # if sigmoid is not None else None
		self.tanh = nn.Tanh()
		self.flatten = nn.Flatten()

		###################################################
		self.supel4_att = QKVAttention(8,self.dec_base_c)
		self.supel3_att = QKVAttention(8,self.dec_base_c//2)
		self.supel2_att = QKVAttention(8,self.dec_base_c//4)
		self.supel1_att = QKVAttention(8,self.dec_base_c//8)
		###################################################
		# self.queel4_att = VKVAttention(self.dec_base_c//2)
		# self.queel3_att = VKVAttention(self.dec_base_c//4)
		# self.queel2_att = VKVAttention(self.dec_base_c//8)
		# self.queel1_att = VKVAttention(self.dec_base_c//32)
		###################################################


	# def get_regressor(self, in_c, out_c, bn = True):
	# 	cls = nn.Sequential(nn.Linear(in_c, in_c),
	# 						nn.ReLU(),
	# 						nn.BatchNorm1d(in_c) if bn else nn.Identity(),
	# 						nn.Linear(in_c, out_c)
	# 						)
	# 	return cls

	def get_encoder(self, use_pretained = False):
		encoder = Encoder(use_pretained)
		return encoder

	# def get_decoder(self, dec_base_c):
	# 	decoder = Decoder(dec_base_c)
	# 	return decoder

	def forward(self, s_imgs, s_msks, q_imgs, q_msks, s_tr_params = None, q_tr_params = None):

		s_imgs = torch.cat([torch.cat(way, dim=0) for way in s_imgs], dim=0)
		s_msks = torch.cat([torch.cat(way, dim=0) for way in s_msks], dim=0)
		q_imgs = torch.cat(q_imgs, dim=0)
		# print(s_imgs.shape, q_imgs.shape, s_msks.shape)
		q_msks = q_msks[:,0,:,:] #torch.cat(q_msks, dim=0)
		# s_tr_params = torch.cat([torch.cat(way, dim=0) for way in s_tr_params], dim=0)
		# q_tr_params = torch.cat(q_tr_params, dim=0)

		s_ifts_l1, s_ifts_l2, s_ifts_l3, s_ifts_l4, _ = self.image_encoder(s_imgs)
		s_mfts_l1, s_mfts_l2, s_mfts_l3, s_mfts_l4, _ = self.mask_encoder(s_msks)
		q_ifts_l1, q_ifts_l2, q_ifts_l3, q_ifts_l4, _ = self.image_encoder(q_imgs)

		#####################################
		#      QUERY MASK PREDICTION        #
		#####################################
		sup4_att1 = self.supel4_att(s_ifts_l4, s_mfts_l4, q_ifts_l4) #2048
		sup3_att1 = self.supel3_att(s_ifts_l3, s_mfts_l3, q_ifts_l3) #1024
		sup2_att1 = self.supel2_att(s_ifts_l2, s_mfts_l2, q_ifts_l2) #512
		sup1_att1 = self.supel1_att(s_ifts_l1, s_mfts_l1, q_ifts_l1) #256

		l4_decout1 = self.el4_dec(self.relu(self.preconv4(self.relu(sup4_att1)))) #1024
		l3_decout1 = self.el3_dec(self.relu(self.preconv3(self.relu(torch.cat([sup3_att1, l4_decout1], dim = 1))))) #512
		l2_decout1 = self.el2_dec(self.relu(self.preconv2(self.relu(torch.cat([sup2_att1, l3_decout1], dim = 1))))) #256
		l1_decout1 = self.el1_dec(self.relu(self.preconv1(self.relu(torch.cat([sup1_att1, l2_decout1], dim = 1))))) #128
		l0_decout1 = self.el0_dec(self.relu(self.preconv0(self.relu(l1_decout1))))

		final_dec_out1 = self.decfin_out(self.relu(l0_decout1))
		final_dec_out1 = self.sigmoid(final_dec_out1/0.01) # Output range : (0, 1)

		# print(final_dec_out.shape, q_msks.shape)
		# print(q_msks.min(), q_msks.max())

		pred_loss1 = self.l2_loss(self.flatten(final_dec_out1), self.flatten(q_msks).float())
		pred_loss2 = self.bce_loss(self.flatten(final_dec_out1), self.flatten(q_msks).float())
		wts = self.flatten(q_msks).clone().float() #.to(dtype = torch.int64)
		# print(np.unique(wts))
		wts[wts > 0.99] = (1-q_msks).sum()/(2*q_msks.sum())
		wts[wts < 1e-4] = 1.0
		# print(np.unique(wts.cpu()))
		pred_loss2 = torch.multiply(wts, pred_loss2).mean()
		# pred_loss3 = self.dice_loss(self.flatten(final_dec_out), self.flatten(q_msks).float())
		pred_loss3 = self.tversky_loss(final_dec_out1, q_msks.long())

		# pred_loss1 = self.l2_loss(self.flatten(final_dec_out[:,1,:,:]), self.flatten(q_msks).float()) \
		# 			+ self.l2_loss(self.flatten(final_dec_out[:,0,:,:]), self.flatten(1 - q_msks).float())
		# pred_loss2 = self.bce_loss(final_dec_out.view((-1,2)), self.flatten(q_msks).long())

		boundary_loss = self.l1_loss(sobel(q_msks[:,None,:,:].float()), sobel((final_dec_out1 > 0.5).float()))


		####################################################################################################

		#######################################
		#      SUPPORT MASK PREDICTION        #
		#######################################
		final_dec_out2 = None
		if not self.is_val:
			s_msks_q = s_msks[:,0,:,:]
			q_msks_s = torch.repeat_interleave((final_dec_out1 > 0.5).to(dtype = torch.float32), repeats = 3, dim = 1)

			q_mfts_l1, q_mfts_l2, q_mfts_l3, q_mfts_l4, _ = self.mask_encoder(q_msks_s)

			sup4_att2 = self.supel4_att(q_ifts_l4, q_mfts_l4, s_ifts_l4) #2048
			sup3_att2 = self.supel3_att(q_ifts_l3, q_mfts_l3, s_ifts_l3) #1024
			sup2_att2 = self.supel2_att(q_ifts_l2, q_mfts_l2, s_ifts_l2) #512
			sup1_att2 = self.supel1_att(q_ifts_l1, q_mfts_l1, s_ifts_l1) #256

			l4_decout2 = self.el4_dec(self.relu(self.preconv4(self.relu(sup4_att2)))) #1024
			l3_decout2 = self.el3_dec(self.relu(self.preconv3(self.relu(torch.cat([sup3_att2, l4_decout2], dim = 1))))) #512
			l2_decout2 = self.el2_dec(self.relu(self.preconv2(self.relu(torch.cat([sup2_att2, l3_decout2], dim = 1))))) #256
			l1_decout2 = self.el1_dec(self.relu(self.preconv1(self.relu(torch.cat([sup1_att2, l2_decout2], dim = 1))))) #128
			l0_decout2 = self.el0_dec(self.relu(self.preconv0(self.relu(l1_decout2))))

			final_dec_out2 = self.decfin_out(self.relu(l0_decout2))
			final_dec_out2 = self.sigmoid(final_dec_out2/0.01) # Output range : (0, 1)

			# print(final_dec_out.shape, q_msks.shape)
			# print(q_msks.min(), q_msks.max())

			pred_loss1 += self.l2_loss(self.flatten(final_dec_out2), self.flatten(s_msks_q).float())
			# pred_loss2 += self.bce_loss(self.flatten(final_dec_out2), self.flatten(s_msks_q).float())
			wts = self.flatten(s_msks_q).clone().float() #.to(dtype = torch.int64)
			# print(np.unique(wts))
			wts[wts > 0.99] = (1-s_msks_q).sum()/(2*s_msks_q.sum())
			wts[wts < 1e-4] = 1.0
			# print(np.unique(wts.cpu()))
			pred_loss2 += torch.multiply(wts, self.bce_loss(self.flatten(final_dec_out2), self.flatten(s_msks_q).float())).mean()
			# pred_loss3 = self.dice_loss(self.flatten(final_dec_out), self.flatten(q_msks).float())
			pred_loss3 += self.tversky_loss(final_dec_out2, s_msks_q.long())

			# pred_loss1 = self.l2_loss(self.flatten(final_dec_out[:,1,:,:]), self.flatten(q_msks).float()) \
			# 			+ self.l2_loss(self.flatten(final_dec_out[:,0,:,:]), self.flatten(1 - q_msks).float())
			# pred_loss2 = self.bce_loss(final_dec_out.view((-1,2)), self.flatten(q_msks).long())

			boundary_loss += self.l1_loss(sobel(s_msks_q[:,None,:,:].float()), sobel((final_dec_out2 > 0.5).float()))

		#####################################
		#     AUGMENTATION PREDICTION       #
		#####################################
		# s_ifts_ap = self.regressor(s_ifts_ap)
		# q_ifts_ap = self.regressor(q_ifts_ap)
		# aug_pred_loss1 = self.l2_loss(s_ifts_ap, s_tr_params)
		# aug_pred_loss2 = self.l2_loss(q_ifts_ap, q_tr_params)
		# # q_tr_params_pred = s_tr_params[:9].view()
		# aug_pred_loss = torch.tensor(0) #aug_pred_loss1 + aug_pred_loss2

		# alignment_loss = F.cosine_similarity(F.adaptive_avg_pool2d(sup4_att, 1).squeeze(),F.adaptive_avg_pool2d(self.supel4_att(q_ifts_l4, l4_decout, s_ifts_l4), 1).squeeze())
		# alignment_loss += F.cosine_similarity(F.adaptive_avg_pool2d(sup3_att, 1).squeeze(),F.adaptive_avg_pool2d(self.supel4_att(q_ifts_l3, l3_decout, s_ifts_l3), 1).squeeze())
		# alignment_loss += F.cosine_similarity(F.adaptive_avg_pool2d(sup3_att, 1).squeeze(),F.adaptive_avg_pool2d(self.supel4_att(q_ifts_l2, l2_decout, s_ifts_l2), 1).squeeze())
		# alignment_loss += F.cosine_similarity(F.adaptive_avg_pool2d(sup1_att, 1).squeeze(),F.adaptive_avg_pool2d(self.supel4_att(q_ifts_l1, l1_decout, s_ifts_l1), 1).squeeze())

		alignment_loss = torch.tensor(0) #1 - alignment_loss/4

		return final_dec_out1, final_dec_out2, pred_loss1, pred_loss2, pred_loss3, boundary_loss, alignment_loss





