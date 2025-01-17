import os
import shutil

filename = '2022-04-18_235027_VID001_Trim'
path = 'outputs/outputs_all-Batch1-28-scheduler/results-videos/video1-frame results/'
newpath = 'outputs/outputs_all-Batch1-28-scheduler/results-videos/new/'

f=os.listdir(path)
f= sorted(f)
i=0
for file in  f:
    filename = file[:-9]
    filename_new = newpath +filename + f'{i:07d}'+'.png'
    shutil.copy(path+file,filename_new)
    i=i+1