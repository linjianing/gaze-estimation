#!/bin/bash
num_person=5
for((i=0; i<$num_person; i=i+1))
do
	python EyeDiapEvaluation.py --out_group_num=$i
done 
