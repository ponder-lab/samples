#!/bin/bash

cmd_name=`basename $0`

hostfile="hostfile"
num_thread="6"

while getopts h:n: opt
do
    case $opt in
        "h" ) flg_h="true" ; hostfile="$OPTARG" ;;
        "n" ) flg_n="true" ; num_thread="$OPTARG" ;;
        * ) echo "Usage: $cmd_name [-h hostfile] [-n num_thread]" 1>&2
            exit 1 ;;
    esac
done

#if [ "$flg_h" = "true" ]; then
#    echo $hostfile
#fi
#
#if [ "$flg_n" = "true" ]; then
#    echo $num_thread
#fi

shift `expr $OPTIND - 1`

if [ $# -ne 1 ]; then
    echo "Usage: $cmd_name [-h hostfile] [-n num_thread]" 1>&2
    exit 1
fi

echo $1
echo $hostfile
echo $num_thread
