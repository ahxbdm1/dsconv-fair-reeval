"""Pre-registered DRIVE-only preprocessing fit, applied unchanged to CHASE."""
import json,sys,time,os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image
from skimage.exposure import equalize_adapthist
import cv2
ROOT=Path(__file__).resolve().parent
SGL=Path(os.environ.get('SGL_DIR',ROOT/'SGL'))/'dataset';DRIVE=Path(os.environ.get('DSCNET_DIR',ROOT/'DSCNet'))/'DSCNet_2D_opensource/Data/DRIVE'
OUT=ROOT/'runs/chase_data'
def atomic(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def pipeline(rgb,c):
    raw=rgb[:,:,1] if c['channel']=='green' else cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    if c['engine']=='skimage':return (equalize_adapthist(raw,clip_limit=c['clip'])*255).astype(np.float32)
    return cv2.createCLAHE(clipLimit=c['clip'],tileGridSize=(8,8)).apply(raw).astype(np.float32)
def fit():
    candidates=[{'channel':'green','engine':'skimage','clip':v} for v in (.008,.009,.010,.011,.012)]
    candidates += [{'channel':channel,'engine':'opencv','clip':v} for channel in ('green','gray') for v in (1,1.5,2,2.5,3)]
    images=sorted((SGL/'DRIVE/HR').glob('*_training.tif'));assert len(images)==20
    pairs=[]
    for p in images:
        rgb=np.array(Image.open(p).convert('RGB'));target=sitk.GetArrayFromImage(sitk.ReadImage(str(DRIVE/'Image_Tr'/(str(int(p.name[:2]))+'.nii.gz')))).astype(np.float32)
        assert rgb.shape[:2]==target.shape,(p,rgb.shape,target.shape)
        pairs.append((rgb,target,p.name))
    scores=[]
    for c in candidates:
        errors=[float(np.mean(np.abs(pipeline(rgb,c)-target))) for rgb,target,_ in pairs]
        scores.append({**c,'mean_absolute_error':float(np.mean(errors)),'per_image_mae':dict(zip([n for _,_,n in pairs],errors))})
    chosen=min(scores,key=lambda r:r['mean_absolute_error'])
    atomic(ROOT/'runs/preprocess_fit.json',{'selection':'minimum mean MAE over 20 DRIVE training images only; scale 0..255; no CHASE labels used','candidates':scores,'selected':chosen})
    for folder,n in [('CHASE',20),('CHASET',8)]:
        images=sorted((SGL/folder/'HR').glob('*.jpg'));assert len(images)==n,(folder,len(images))
        image_dest=OUT/('Image_Tr' if folder=='CHASE' else 'Image_Te');label_dest=OUT/('Label_Tr' if folder=='CHASE' else 'Label_Te')
        image_dest.mkdir(parents=True,exist_ok=True);label_dest.mkdir(parents=True,exist_ok=True)
        for p in images:
            name=p.stem;rgb=np.array(Image.open(p).convert('RGB'))
            label=np.array(Image.open(SGL/folder/'VE'/(name+'_1stHO.png')).convert('L'))>127
            sitk.WriteImage(sitk.GetImageFromArray(pipeline(rgb,chosen)),str(image_dest/(name+'.nii.gz')))
            sitk.WriteImage(sitk.GetImageFromArray(label.astype(np.float32)),str(label_dest/(name+'.nii.gz')))
    prepare_split();atomic(ROOT/'runs/chase_prepare_done.json',{'time':time.time(),'selected':chosen,'raw_train':20,'validation':4,'test':8})
def prepare_split():
    dest=OUT/'protocol';dest.mkdir(parents=True,exist_ok=True)
    alltr=sorted((OUT/'Image_Tr').glob('*.nii.gz'));te=sorted((OUT/'Image_Te').glob('*.nii.gz'))
    assert len(alltr)==20 and len(te)==8
    val=[alltr[i] for i in (0,5,10,15)];tr=[p for p in alltr if p not in val]
    split={'train':[p.name for p in tr],'val':[p.name for p in val],'test':[p.name for p in te]}
    atomic(ROOT/'runs/split_chase.json',split)
    for name,im in [('original_train',alltr),('clean_train',tr),('val',val),('test',te)]:
        (dest/(name+'.txt')).write_text('\n'.join(map(str,im))+'\n')
        labels=[OUT/('Label_Te' if name=='test' else 'Label_Tr')/p.name for p in im]
        (dest/(name+'_labels.txt')).write_text('\n'.join(map(str,labels))+'\n')
        if not (dest/(name+'.npy')).exists():
            arrays=[sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32) for p in im]
            count=sum(x.size for x in arrays);mean=0.;var=0.
            for x in arrays:mean+=np.sum(x)
            mean/=count
            for x in arrays:var+=np.sum(np.square(x-mean))
            np.save(dest/(name+'.npy'),[mean,np.sqrt(var/count)])
    return dest
if __name__=='__main__':fit()
