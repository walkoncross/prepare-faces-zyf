#!/bin/sh

declare -x GLOG_minloglevel=2

splits=32

let cnt=splits-1
for i in `seq 0 $cnt`;
do
	echo 'loop-'$i
	nohup ./vggface2_test_mtcnn_align_crop_96x112_zyf.py $splits $i > '/workspace/vgg2-test-process-log-'$splits'-'$i'.txt' &
done

