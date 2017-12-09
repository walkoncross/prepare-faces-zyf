import os.path as osp
import numpy as np

from numpy.linalg import norm

fn_list = 'picked_100_ids_images.txt'
feat_root_dir = './msceleb-random-100ids-feat'
top_n = 10

base_name = osp.basename(fn_list)
fn_rlt = osp.splitext(base_name)[0] + ('_top_%d_sim2center.txt' % top_n)

fp_list = open(fn_list, 'r')
fp_rlt = open(fn_rlt, 'w')

last_idx = -1

img_fn_list = []
feat_list = []

for line in fp_list:
    if not line.strip():
        continue

    spl = line.split()
    img_fn = spl[0]
    idx = int(spl[-1])

    feat_fn = osp.splitext(img_fn)[0] + '.npy'
    feat = np.load(osp.join(feat_root_dir, feat_fn))
    feat /= norm(feat)

    if last_idx < 0 or idx == last_idx:
        feat_list.append(feat.tolist())
        img_fn_list.append(line)
    else:
        id_name = img_fn_list[0].split('/', 1)[0]
        print '===> id_name: %s \n' % id_name

        top_n_t = min(top_n, len(feat_list))
        fp_rlt.write('id_name: %s\n' % id_name)
        # fp_rlt.write('top: %d\n' % top_n_t)

        feat_list_mat = np.array(feat_list)
        feat_center = np.mean(feat_list_mat, 0)

        sim_mat = np.dot(feat_list_mat, feat_center)
        sort_idx = np.argsort(-sim_mat)
        # top_n_idx = sort_idx[:top_n_t]
        # print 'top %d sorted similarity with feat_center: ' % (top_n_t)
        # print top_n_idx

        cnt = 0
        tmp_list = []
        for i in sort_idx:
            if img_fn_list[i] not in tmp_list:
                tmp_list.append(img_fn_list[i])
                cnt += 1
                print '--> %s %4.3f' % (img_fn_list[i], sim_mat[i])
                fp_rlt.write('%s %4.3f\n' % (img_fn_list[i], sim_mat[i]))
                if cnt == top_n_t:
                    break

        feat_list = []
        img_fn_list = []
        feat_list.append(feat.tolist())
        img_fn_list.append(line)

    last_idx = idx

fp_list.close()
fp_rlt.close()
