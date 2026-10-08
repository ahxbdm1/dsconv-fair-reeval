import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_clean as t
import numpy as np
read=t.sitk.ReadImage
paths=list((t.DATA/'Image_Tr').glob('*.nii.gz'))+list((t.DATA/'Label_Tr').glob('*.nii.gz'))
paths+=list((t.ROOT/'runs/chase_data/Image_Tr').glob('*.nii.gz'))+list((t.ROOT/'runs/chase_data/Label_Tr').glob('*.nii.gz'))
uncached={str(p):t.sitk.GetArrayFromImage(read(str(p))) for p in paths}
from performance_cache import enable
enable()
for p in paths:
    for _ in range(2):assert np.array_equal(uncached[str(p)],t.sitk.GetArrayFromImage(t.sitk.ReadImage(str(p))))
print('Source cache arrays bitwise identical for',len(paths),'images and labels')
