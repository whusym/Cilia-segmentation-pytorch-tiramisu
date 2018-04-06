#!/bin/bash

# script for running Beat Frequency filter.
trainPath=<trainPath>
testPath=<testPath>
filePath=<filePath>

while read line
	do
		cd $trainPath/$line
		python BeatFrequency.py $line
	done<$filePath/train.txt

while read line
	do
		cd $testPath/$line
		python BeatFrequency.py $line
	done<$filePath/test.txt
