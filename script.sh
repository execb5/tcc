#! /usr/local/bin/zsh

files=(input/photos/*)
for i in `seq 10`
do
	time python -O src/main.py ${files[RANDOM % ${#files[@]}]}
done

#for i in `seq 10`
#do
	#time python -O src/main.py input/photos/IXJ-6605.JPG
#done
