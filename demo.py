from thop import profile
import torch
from SFSL import SFSL_SiameseNet
from cnn_feat import CNN_FEAT_SiameseNet
from cdnet import cdnet_SiameseNet
from cosimnet import deeplab_SiameseNet
from cdnet_fcn import FCN8_SiameseNet
from dof_cdnet import dof_cdnet_SiameseNet
from cscdnet import CSCDNet_arch
from LSGCANet_BAM import LSGCANet_BAM_SiameseNet
from LSGCANet_SAB import LSGCANet_SAB_SiameseNet
from LSGCANet_CBAM import LSGCANet_CBAM_SiameseNet
from LSGCANet import LSGCANetSiameseNet
from SimSaC_m import SimSaC_Model

input1 = torch.randn(1, 3, 512, 512)
input2 = torch.randn(1, 3, 512, 512)

print("##########################")
print("Model Complexity of SFSL")
model = SFSL_SiameseNet()
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of LSGCANet-BAM")
model = LSGCANet_BAM_SiameseNet()
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of LSGCANet-SAB")
model = LSGCANet_SAB_SiameseNet()
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of LSGCANet-CBAM")
model = LSGCANet_CBAM_SiameseNet()
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of LSGCANet")
model = LSGCANetSiameseNet()
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of CNN-FEAT")
model = CNN_FEAT_SiameseNet()
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of CDNET")
model = cdnet_SiameseNet()
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of CDNET-FCN")
model = FCN8_SiameseNet()
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of CosimNet")
model = deeplab_SiameseNet()
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of DOF-CDNet")
model = dof_cdnet_SiameseNet()
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of CSCDNet")
model = CSCDNet_arch(outc=2)
macs, params = profile(model, inputs=(input1,input2))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")

print("##########################")
print("Model Complexity of SimSaC")
model = SimSaC_Model(batch_norm=True, pyramid_type='VGG',
                         div=1.0, evaluation=False,
                         consensus_network=False,
                         cyclic_consistency=True,
                         dense_connection=True,
                         decoder_inputs='corr_flow_feat',
                         refinement_at_all_levels=False,
                         refinement_at_adaptive_reso=True,
                         num_class=2,
                         use_pac = False,
                         vpr_candidates=False)

input1_reshape = torch.randn(1, 3, 256, 256)
input2_reshape = torch.randn(1, 3, 256, 256)

macs, params = profile(model, inputs=(input1,input2,input1_reshape,input2_reshape))
print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
print("##########################")
