#!/bin/sh

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
unzip LD2011_2014.txt.zip
rm LD2011_2014.txt.zip
rm -rf __MACOSX

mkdir data
mv LD2011_2014.txt data
