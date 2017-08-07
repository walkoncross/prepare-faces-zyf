import sys
import os
import os.path as osp

f = open('/disk2/data/FACE/adience/fold_3_data.txt','r')
lines = f.readlines()

fp = open('adience-test-3.txt','w')

for line in lines[1:]:
    #print line
    splits = line.split('\t')
    filename = 'coarse_tilt_aligned_face.' + splits[2] + '.' + splits[1]
    filepath = osp.join('adience/faces',splits[0],filename)
    print filepath
    if splits[4]=='m':
        fp.write('{} {}\n'.format(filepath, '0'))
        print '0'
    elif splits[4]=='f':
        fp.write('{} {}\n'.format(filepath, '1'))
        print '1'
    else:
        continue
fp.close()
