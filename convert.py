import json
import cv2
import numpy as np
import os
data_dir = '/data/mc_data/select/'
ins_list = []
process_list = ['label_data_0601.json','label_data_0313.json','label_data_0531.json']
with open(os.path.join(data_dir,'val.txt'),'r') as f:
    for line in f: 
        image,bin_label,ins_label = line.strip().split(" ")
        ins_list.append(ins_label.split("/")[-3]+'_'+ins_label.split("/")[-2]+'_'+ins_label.split("/")[-1][:-4])

gt = []


for js in process_list:
    with open(data_dir+js,"r") as f:
        for line in f:
            tmp= json.loads(line.strip("\n"))
            file = tmp['raw_file']
            key = file.split("/")[-3]+'_'+file.split("/")[-2]+'_'+file.split("/")[-1][:-4]
            
            if key in ins_list:
                lb = {}
                lb['image'] = key
                lb['lanes'] = []
                x_values = tmp['lanes']
                for ll in x_values:
                    add = []
                    for i,x in enumerate(ll):
                        if (x!=-2):
                           add.append({'y':tmp['h_samples'][i],'x':x})
                    lb['lanes'].append(add)
                gt.append(lb)

with open('tusimple_gt.json','w') as f:
    json.dump(gt,f, indent = 4)