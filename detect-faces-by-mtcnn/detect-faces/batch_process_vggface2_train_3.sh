#!/bin/sh

declare -x GLOG_minloglevel=2

splits=96

#let cnt=splits-1
start=64

let end=start+31

for i in `seq $start $end`;
do
	echo 'loop-'$i
	nohup ./vggface2_mtcnn_align_crop_96x112_zyf.py $splits $i > '/workspace/process-log-'$splits'-'$i'.txt' &
done

