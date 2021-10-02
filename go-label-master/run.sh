#!/bin/bash
server=$1
root=$2
file=$3
if [ -z $server ]
then 
    echo "Error: ip address not set"
    exit
fi
if [[ $server =~ (.*)\:(.*) ]]
then
    echo $server
else
    echo "Error: pass valid ip:port"
    exit
fi
if [ -z $root ]
then 
    root=/ssd
fi
if [ -z $file ]
then 
    file=empty
fi
 
/etc/init.d/nginx reload
/etc/init.d/nginx restart
./label --server="http://${server}" --root="${root}" --file="${file}"
