#!/bin/bash

while read line
	do 
		python masks.py masks/$line
	done<train.txt