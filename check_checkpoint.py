import torch

checkpoint_path = "save/CI4R/classify/CL_S_0_M_RADMAMBA_B_16_LR_0.0050_H_64_P_47865_FL_224_ST_1.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")
    if isinstance(checkpoint[key], dict):
        print(f"    (dict with {len(checkpoint[key])} items)")
    elif isinstance(checkpoint[key], torch.Tensor):
        print(f"    (tensor: {checkpoint[key].shape})")
    else:
        print(f"    ({type(checkpoint[key]).__name__})")
