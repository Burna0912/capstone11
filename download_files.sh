#!/bin/bash

mkdir -p data/original/images
mkdir -p data/original/labels

while read fname; do
  echo "Downloading: $fname"
  scp -P 1004 ubuntu@203.253.70.233:/home/ubuntu/mydata/capstone/val/images/"$fname" ./data/original/images/
  labelname="${fname%.jpg}.txt"
  scp -P 1004 ubuntu@203.253.70.233:/home/ubuntu/mydata/capstone/val/labels/"$labelname" ./data/original/labels/
done < filelist.txt
