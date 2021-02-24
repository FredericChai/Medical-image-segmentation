import torch
import torch.nn.functional as F

from dice_loss import dice_coeff
import matplotlib.pyplot as plt 
import os
from torchvision.utils import save_image
from shutil import copyfile
from sklearn.metrics import recall_score,jaccard_score,accuracy_score,precision_score
#Use dice_loss to evaluate the performance 
def eval_net(net, dataset,current_epoch, gpu,save_sample_mask,result_path):
	"""Evaluation without the densecrf with the dice coefficient"""
	net.eval()
	tot = 0
	#dataset is the dataloader['val']
	with torch.no_grad():
		for i,data in enumerate(dataset):
			img = data[0]
			true_mask = data[1]
			img_ids = data[2]

			left_img = img['left']
			right_img = img['right']

			if gpu:
				left_img = left_img.cuda()
				right_img = right_img.cuda()
				true_mask = true_mask.cuda()

			left_mask_pred = net(left_img.unsqueeze(dim=0))
			left_mask_pred = F.upsample(left_mask_pred,size=[512,512],mode = 'bicubic')
			right_mask_pred = net(right_img.unsqueeze(dim=0))
			right_mask_pred = F.upsample(right_mask_pred,size=[512,512],mode = 'bicubic')
			#channel,height must be same as mask_pred,while width need to be same as true_mask
			batch = left_mask_pred.shape[0]
			channel = left_mask_pred.shape[1]
			height,width = true_mask.shape[1],true_mask.shape[2]
			plug = torch.zeros(batch,channel,height,width-512).cuda()
			# left_mask_pred, right_mask_pred = left_mask_pred.cpu(),right_mask_pred.cpu()
			left_mask_pred = torch.cat((left_mask_pred,plug),3)
			right_mask_pred = torch.cat((plug,right_mask_pred),3)
			mask_pred = left_mask_pred+right_mask_pred
			_, mask_pred = torch.max(mask_pred, dim=1)
			# mask_pred = torch.unsqueeze(mask_pred, dim=1)
			#mask_pred = F.upsample(mask_pred,size=true_mask.shape,mode = 'bicubic')
			# true_mask = true_mask.squeeze(dim=1)
			#mask_pred = (mask_pred > 0.5).float()
			dice = dice_coeff(mask_pred.float(), true_mask.float()).item()
			tot += dice

			if save_sample_mask ==True:
				save_mask(i,mask_pred,true_mask,current_epoch,img_ids,dice,result_path)
	return tot / (i + 1)

def test_net(net,dataset,checkpoint, gpu,result_path):
	"""evaluate model with test data"""
	#load best checkpoint
	net.load_state_dict(torch.load(checkpoint))
	net.eval()
	total_dice = 0
	# total_dice = 0
	total_accuracy = 0
	total_sensitivity = 0
	total_jaccard = 0
	total_precision = 0
	#load test data with original size mask and resized img
	for i ,data in enumerate(dataset):
		img = data[0]
		resized_mask = data[1]
		original_mask = data[2]
		img_ids = data[3]

		left_img = img['left']
		right_img = img['right']

		if gpu:
			left_img = left_img.cuda()
			right_img = right_img.cuda()
			resized_mask = resized_mask.cuda()
			original_mask = original_mask.cuda()

		left_mask_pred = net(left_img.unsqueeze(dim=0))
		
		right_mask_pred = net(right_img.unsqueeze(dim=0))
		

		#channel,height must be same as mask_pred,while width need to be same as true_mask
		batch,channel = left_mask_pred.shape[0],left_mask_pred.shape[1]
		height,width = resized_mask.shape[1],resized_mask.shape[2]
		left_mask_pred = F.upsample(left_mask_pred,size=[height,512],mode = 'bicubic')
		right_mask_pred = F.upsample(right_mask_pred,size=[height,512],mode = 'bicubic')
		plug = torch.zeros(batch,channel,height,width-512).cuda()
		# left_mask_pred, right_mask_pred = left_mask_pred.cpu(),right_mask_pred.cpu()
		left_mask_pred = torch.cat((left_mask_pred,plug),3)
		right_mask_pred = torch.cat((plug,right_mask_pred),3)

		mask_pred = left_mask_pred+right_mask_pred
		_, mask_pred = torch.max(mask_pred, dim=1)
		#the mask_pred will become [1,512,682] after max manipulation
		# mask_pred = torch.unsqueeze(mask_pred, dim=1)
		mask_pred = mask_pred.float().unsqueeze(dim=0)
		org_h,org_w = original_mask.shape[1],original_mask.shape[2]
		mask_pred = F.upsample(mask_pred,size=(org_h,org_w),mode = 'bilinear')
		# true_mask = true_mask.squeeze(dim=1)
		mask_pred = (mask_pred > 0.5).float()
		dice = dice_coeff(mask_pred.squeeze(dim=0), original_mask).item()
		total_dice += dice
		print('img:',img_ids,'dice:',dice,'numer:',i)
		mask_pred = mask_pred.view(-1).cpu().numpy().astype('int')
		original_mask = original_mask.view(-1).cpu().numpy().astype('int')

		total_accuracy += accuracy_score(mask_pred, original_mask)
		total_sensitivity += recall_score(mask_pred, original_mask)
		total_jaccard += jaccard_score(mask_pred, original_mask)
		total_precision += precision_score(mask_pred, original_mask)
		# save_test_result(i,mask_pred,original_mask,img_ids,dice,target_path)
	avg_dice = total_dice/(i+1)
	avg_acc  =total_accuracy/(i+1)
	avg_sens = total_sensitivity/(i+1)
	avg_jaccard = total_jaccard/(i+1)
	avg_specificity = total_precision/(i+1)
		# save_test_result(i,mask_pred,original_mask,img_ids,dice)
	avg_dice = total_dice/(i+1)
	print('avg_dice:',avg_dice)
	#can add sensitivity, accuracy later
	writer = open(result_path+'/STD_trained_Test_result.txt','w')
	writer.write('average_dice:'+str(avg_dice)+'\n'+
				 'average_jaccard:'+str(avg_jaccard)+'\n'+
				 'average_accuracy:'+str(avg_acc)+'\n'+
				 'average_sensitivity:'+str(avg_sens)+'\n'+
				 'average_specificity:'+str(avg_specificity)+'\n')
	return

def save_mask(index,pred,ground_truth,current_epoch,img_ids,dice,result_path):
	"""save sample image for validation"""
	path = result_path+ '/validate_mask/'+str(current_epoch)+'/'
	if not os.path.exists(path):
		os.makedirs(path)
	# pred_im = pred.cpu().numpy()
	# ground_truth_im = ground_truth.cpu().numpy
	save_image(pred,path+str(img_ids)+'_pred_'+str(dice)+'_.png')
	save_image(ground_truth,path+str(img_ids)+'_groundTruth_'+str(dice)+'_.png')
	# for i in img_ids:
	#     copyfile('/mnt/HDD2/Frederic/seg_optimization/data/resized_ISIC2016/Val_Data/'+i,path+i)

def save_test_result(index,pred,ground_truth,img_ids,dice,result_path):
	"""save sample image for validation"""
	path = result_path+'/test_result/'
	if not os.path.exists(path):
		os.mkdir(path)
	# pred_im = pred.cpu().numpy()
	# ground_truth_im = ground_truth.cpu().numpy
	save_image(pred,path+str(img_ids)+'_pred_'+str(dice)+'_.png')
	save_image(ground_truth,path+str(img_ids)+'_groundTruth_'+str(dice)+'_.png')
	# for i in img_ids:
	#     copyfile('/mnt/HDD2/Frederic/seg_optimization/data/resized_ISIC2016/Val_Data/'+i,path+i)
