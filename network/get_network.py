#from network import ResNet
import sys
sys.path.append("/data3/xytan/Fedavg_DropPos/data3/xytan/Fedavg_DropPos/network")
#import network_
#from .byol import BYOL
#from network_.backbone.resnet import resnet50
from .models_DropPos_mae import DropPos_mae_vit_base_patch16_dec512d8b, DropPos_mae_vit_small_patch16_dec512d2b

"""
def GetNetwork_global(args, num_classes, **kwargs):
    if args.model == 'deeplabv3plus_resnet50':
        #model = ResNet.resnet18(pretrained=pretrained, num_classes=num_classes, **kwargs)
        #feature_level = 512
        model = network_.deeplabv3plus_resnet50(num_classes=num_classes, output_stride=args.output_stride)

    else:
        raise ValueError("The model is not support")

    #return model, feature_level
    return model
"""


def GetNetwork():

    #resnet = resnet50(pretrained=False)
    #model = BYOL(resnet, image_size=512, hidden_layer="avgpool")

    """
    model = models_DropPos_mae.__dict__['DropPos_mae_vit_base_patch16'](norm_pix_loss=True,
                                                        mask_token_type='param',
                                                        shuffle=False,
                                                        multi_task=False,
                                                        conf_ignore=False,
                                                        attn_guide=True)
    """
    
    model = DropPos_mae_vit_base_patch16_dec512d8b(norm_pix_loss=True,
                                                        mask_token_type='param',
                                                        shuffle=False,
                                                        multi_task=False,
                                                        conf_ignore=False,
                                                        attn_guide=True)
    
    
    """
    model = DropPos_mae_vit_small_patch16_dec512d2b(norm_pix_loss=True,
                                                        mask_token_type='param',
                                                        shuffle=False,
                                                        multi_task=False,
                                                        conf_ignore=False,
                                                        attn_guide=True)
    """
    

    #return model, feature_level
    return model
