#!/bin/bash

# 
for i in ../Test/*.ipynb; do
	echo $i
	clear_notebook $i
	# break
done 
