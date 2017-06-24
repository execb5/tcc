#! /usr/local/bin/zsh

#files=(input/photos/*)
#for i in `seq 10`
#do
	#time python -O src/main.py ${files[RANDOM % ${#files[@]}]}
#done

#for i in `seq 10`
#do
	#time python -O src/main.py input/photos/IXJ-6605.JPG
#done

for i in `seq 10`
do
	time python -O src/main.py input/photos/IPN-5186.JPG input/photos/ITA-0204.JPG input/photos/IUZ-5286.JPG input/photos/IVA-5837.JPG input/photos/IVG-3501.JPG input/photos/IWS-5800.JPG input/photos/IWU-3070.JPG input/photos/IXJ-6605.JPG input/photos/IYA-3344.JPG input/photos/OQF-8425.JPG
done
