# Copyright (c) 2025 Yaolei Qi
# Copyright (c) 2026 Heng Zhang
# SPDX-License-Identifier: MIT
# Adapted from DSCNet; see THIRD_PARTY_NOTICES.md.
"""Exact upstream tile coordinates, weighting and IO; batch tiles per image."""
from pathlib import Path
import numpy as np,torch,SimpleITK as sitk
from S3_Train_Process import reshape_img
def infer_batch8(model,data,subset,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);model.eval();roi=224
    mean,std=np.load(data/(subset+'.npy'))
    for source in (data/(subset+'.txt')).read_text().splitlines():
        image=sitk.GetArrayFromImage(sitk.ReadImage(source)).astype(np.float32);image=(image-mean)/std
        y,x=image.shape;ind=(max(y,x)//roi+1)*roi
        image=reshape_img(image,ind,ind)[None,None]
        pred=np.zeros((1,1,ind,ind),dtype=np.float32);n_map=np.zeros_like(pred)
        a=np.where(np.zeros((roi,roi))==0)
        with np.errstate(divide='ignore'):kernel=(1/((a[0]-roi//2)**4+(a[1]-roi//2)**4)).reshape(1,roi,roi)
        coords=[];stride=roi//2
        for i in range(y//stride):
            for j in range(x//stride):coords.append((i*stride,j*stride))
            coords.append((i*stride,y-roi))
        for j in range(x//stride-1):coords.append((x-roi,j*stride))
        coords.append((x-roi,y-roi))
        with torch.no_grad():
            for k in range(0,len(coords),8):
                part=coords[k:k+8]
                tiles=np.concatenate([image[:,:,u:u+roi,v:v+roi] for u,v in part],axis=0)
                output=model(torch.from_numpy(tiles).cuda()).data.cpu().numpy()
                for index,(u,v) in enumerate(part):
                    pred[:,:,u:u+roi,v:v+roi]+=output[index:index+1]*kernel
                    n_map[:,:,u:u+roi,v:v+roi]+=kernel
        with np.errstate(invalid='ignore',divide='ignore'):result=(pred/n_map).astype(np.float32)
        sitk.WriteImage(sitk.GetImageFromArray(result[0,0,:y,:x]),str(out/Path(source).name))
