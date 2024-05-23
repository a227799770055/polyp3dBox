import os, sys
sys.path.append('/home/insign2/work/flexible-yolov5')
import torch
from torchvision.models import resnet18  # 這裡以 ResNet-18 為例
from thop import profile

device = 'cuda'
checkpoint_list = ['polyp_morph_效能驗證_0619', 'test_yolo_0810','test_resnet34_0810','polyp_morresnet18_效能驗證_0814',
                    'polyp_morphyoloNew_效能驗證_0816','polyp_morphDeep_效能驗證_0904']
for checkpoint in checkpoint_list:
    model_checkpoint = './Polyp/{}/weights/best.pt'.format(checkpoint)
    model = torch.load(model_checkpoint)['model']
    model = model.to(device).float()
    input_data = torch.randn(8, 3, 512, 512).to('cuda')  # 假設輸入尺寸為 (1, 3, 224, 224)
    flops, params = profile(model, inputs=(input_data,))
    print('======================')
    print(checkpoint)
    print(f"params: {params} ")
    print(f"FLOPs: {flops / 1e9} G")

