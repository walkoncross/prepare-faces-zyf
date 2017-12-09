import random
import os.path as osp

total_ids = 10000
n_pick = 10

fn_list = ''

fn_picked_ids = './picked_%d_ids.txt' % n_pick
fn_picked_imgs = './picked_%d_ids_images.txt' % n_pick

picked_ids = random.sample(xrange(total_ids), n_pick)
picked_ids = sorted(picked_ids)

print "%d ids picked from %d ids: " % (n_pick, total_ids)
print '\t', picked_ids

print 'save picked ids into file: ', osp.abspath(fn_picked_ids)

fp = open(fn_picked_ids, 'w')
for i in picked_ids:
    fp.write('%d\n' % i)
fp.close()

if fn_list:
    print 'save images of picked ids into file: ', osp.abspath(fn_picked_imgs)
    
    fp_list = open(fn_list, 'r')
    fp_rlt = open(fn_picked_imgs,'w')

    for line in fp_list:
        if not line.strip():
            continue

        idx = line.split()[-1]

        if int(idx) in picked_ids:
            fp_rlt.write(line)

    fp_list.close()
    fp_rlt.close()
