#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 01 06:36:05 2017

@author: zhaoy
"""

fn = './landmark.json'
fn_splits = fn.rsplit('.', 1)
fn_out = fn_splits[0] + '_correct.' + fn_splits[1]

fp_in = open(fn, 'r')
fp_out = open(fn_out, 'w')

for line in fp_in:
    line_new = line.replace('}{', '}, {')
    line_new = line_new.replace('} {', '}, {')
    fp_out.write(line_new)

fp_in.close()
fp_out.close()