#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 2 2017

@author: zhaoy
"""

import sys


def main(list_fn):
    start_id = -1
    cnt = 0

    dup_cnt = 0

    fp = open(list_fn, 'r')

    tmp_list = []

    for line in fp:
        spl = line.strip().split()
        idx = int(spl[1])

        cnt += 1

        if idx != start_id:
            tmp_list = []
            start_id = idx
        else:
            if line in tmp_list:
                dup_cnt += 1

        tmp_list.append(line)

        print(
            '\n---> processed {} image file name with {} old ids\n'.format(cnt - 1, start_id + 1))
        print('\t found {} duplicated lines\n'.format(dup_cnt))

    print(
        '\n\n===> processed {} image file name with {} old ids\n'.format(cnt - 1, start_id + 1))
    print('\t found {} duplicated lines\n'.format(dup_cnt))

    fp.close()


if __name__ == '__main__':
    list_fn = ''
    if len(sys.argv) > 1:
        list_fn = sys.argv[1]

    main(list_fn)
