#!/cs/local/bin/bash

# Author      : Mingbin Xu (mingbin.xu@gmail.com)
# Filename    : prepare.sh
# Last Update : Feb 2, 2016
# Description : The entry point of LM training
# Website     : https://wiki.eecs.yorku.ca/lab/MLL/

# Copyright (c) 2016 iNCML (author: Mingbin Xu)
# License: MIT License (see LICENSE)

# Reset
Color_Off="\033[0m"       # Text Reset

# Regular Colors
Red="\033[0;31m"          # Red


if [ $# -ne 3 ] 
then
	printf ${Red}
	printf "usage: %s <basename> <min-count> <max-size>\n" $0 1>&2
	printf "    <basename>  : basename of the data set, e.g. ptb \n" 1>&2
	printf "    <min-count> : words whose occurrence is less than min-count are mapped to <unk>\n" 1>&2
	printf "    <max-size>  : the vocabulary size is limited to max-size\n" 1>&2
	printf ${Color_Off}
	exit 1
fi

basename=${1}
min_count=${2}
max_size=${3}


for f in "config" "numeric-data" "raw-data" "source" "model-archive"
do
	if [ ! -d $f ] 
	then 
		printf ${Red}
		printf "Please organize the files according to README. \n"
		printf ${Color_Off}
		exit 1
	fi
done


if [ ! -f "trainer" ]
then
	cd ./source
	nvcc -O3 -arch=sm_30 -lcurand -lcublas -Xcompiler -rdynamic -o ../trainer trainer.cpp network.cpp layer.cpp batch-constructor.cpp matrix.cu
	if [ $? -ne 0 ]
	then 
		printf ${Red}
		printf "Fail to compile trainer. \n"
		printf ${Color_Off}
		exit 1
	fi
	cd ..
	echo "traier is compiled"
fi


if [ ! -f "vocabulary" ]
then
	cd ./source
	g++ -o ../vocabulary vocabulary.cpp -O3 -rdynamic
	if [ $? -ne 0 ]
	then 
		printf ${Red}
		printf "Fail to compile vocabulary. \n"
		printf ${Color_Off}
		exit 1
	fi
	cd ..
	echo "vocabulary is compiled"
fi


if [ ! -f "numericize" ]
then
	cd ./source
	g++ -o ../numericize numericize.cpp -O3 -rdynamic
	if [ $? -ne 0 ]
	then 
		printf ${Red}
		printf "Fail to compile trainer. \n"
		printf ${Color_Off}
		exit 1
	fi
	cd ..
	echo "numericize is compiled"
fi


rm -rf ${basename}.vocab
vocabulary "raw-data/"${basename}.train.txt ${min_count} ${max_size} > ${basename}.vocab
if [ $? -ne 0 ]
then 
	printf ${Red}
	printf "Fail to collect vocabulary statistics. \n"
	printf ${Color_Off}
	exit 1
fi
echo "vocabulary statistics is collected"


for data in `ls raw-data`
do
	numericize ${basename}.vocab "raw-data/"${data} "numeric-data/"`basename ${data} .txt`.numeric
	if [ $? -ne 0 ]
	then 
		printf ${Red}
		printf "Fail to numericize %s. \n" ${data}
		printf ${Color_Off}
		exit 1
	fi
done
echo "data set has been numericized"

echo "preparation done:)"
