
import torch 
from torchvision import models
from resnet50cds import ResNet50
from efficientnet_v2 import EfficientNetV2
import math    
#from coatnet import coatnet_4

from timm.models.vision_transformer import vit_small_patch16_224


def get_model(name="vgg16", pretrained=True):
	if name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
	elif name == "resnet50":
		model = models.resnet50(pretrained=pretrained)
	elif name == "vision_transformer":
		model = vit_small_patch16_224(pretrained=True,pretrained_cfg_overlay=dict(file='/home/cds/.cache/torch/hub/checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'))
		model.head = torch.nn.Linear(model.head.weight.shape[1], 4)
	elif name == "resnet50cds":
		model = ResNet50(image_depth=3, num_classes=4, use_cbam=False)
	# elif name == "coatnet":
	# 	model = coatnet_4()
	elif name == "EfficientNetV2":
		model = EfficientNetV2('s',in_channels=3,n_classes=4,pretrained=False)
	# elif name == "resnet50cds":
    # 	model = ResNet50(image_depth=1, num_classes=4, use_cbam=True)
	elif name == "densenet121":
		model = models.densenet121(pretrained=pretrained)		
	elif name == "alexnet":
		model = models.alexnet(pretrained=pretrained)
	elif name == "vgg16":
		model = models.vgg16(pretrained=pretrained)
	elif name == "vgg19":
		model = models.vgg19(pretrained=pretrained)
	elif name == "inception_v3":
		model = models.inception_v3(pretrained=pretrained)
	elif name == "googlenet":		
		model = models.googlenet(pretrained=pretrained)
	
		
	if torch.cuda.is_available():
		return model.cuda()
		model=torch.nn.DataParallel(model,device_ids=[0,1,2,3,4,5])
		model.cuda('cuda:0,1,2,3')
	else:
		return model 
def model_norm(model_1, model_2):
	squared_sum = 0
	for name, layer in model_1.named_parameters():
	#	print(torch.mean(layer.data), torch.mean(model_2.state_dict()[name].data))
		squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
	return math.sqrt(squared_sum)