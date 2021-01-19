#!/bin/sh

gender='K'

if [ "$1" != "" ]
then
    gender="$1"
fi

for f in `ls trainall/*_$gender.wav`
do
    res=`python prog.py "$f"`
    printf "$res "
    printf "$f" 1>&2
    echo
done