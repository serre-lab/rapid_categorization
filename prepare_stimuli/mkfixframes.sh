#!/bin/bash
# Create symlink-copies of fixation frame to be read by ffmpeg video generation

if [ ! -h "tmp/img0000.png" ]; then
	mkdir tmp/
	for i in $(seq -w 0000 1000); do
		ln -s $(pwd)/fix.png "tmp/img${i}.png"
		echo $i
	done
else
	echo "Fix frames already exist."
fi


