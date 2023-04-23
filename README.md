  
## How to use


python demo.py

* Basic usage 
    ```python
    from LSGCANet import LSGCANetSiameseNet
    from thop import profile
    model = LSGCANetSiameseNet()
    macs, params = profile(model, inputs=(input1,input2))
    print("macs is {} G,params is {} M".format(macs / (1000 ** 3), params / (1000 ** 2)))
    ```  
## Run

![image](https://github.com/guocheer/Macs_SCD/blob/main/Macs.png)

## Acknowledgement

We heavily borrow code from public projects, such as [Macs](https://github.com/Lyken17/pytorch-OpCounter), [SimSaC](https://github.com/SAMMiCA/SimSaC), [CSCDNet](https://github.com/kensakurada/sscdnet)...
  
## Results of Scene Change Detection

<p align="center">
<table>
<tr>
<td>

Model | Params(M) | MACs(G)
---|---|---
CNN-Feat | 2.33 | 7.05
CDNet | 0.77 | 68.81
CDNet-FCN | 134.26 | 380.05
DOF-CDNet | 75.38 | 206.06
CosimNet | 38.31 | 407.53
CSCDNet | 92.30 | 75.94
HPCFNet | 35.24 | 138.62
SimSac | 18.15 | 193.48
LSGCANet | 47.43 | 164.37

</td>
<td>
    
Model | Params(M) | MACs(G)
---|---|---
Baseline SFSL | 44.74 | 54.04
SFSL-CBAM | 46.51 | 135.12
SFSL-SAB | 47.15 | 141.23
SFSL-BAM | 46.84 | 149.45
SFSL-LSGCAM | 47.43 | 164.37

</td>
</tr>
</p>
