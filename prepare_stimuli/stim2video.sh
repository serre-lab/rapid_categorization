#!/bin/bash
# Generate a .webm video from an input image for rapid presentation by adding frames of fix.png before and after

# go to script path
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Parameter checks
if [ $# -lt 4 ]; then
	echo "Usage: $0 input_image output_video fix_time_ms after_time_ms fps"
	echo "  e.g.: $0 scene.png scene.webm 1000 500 60"
	exit 1
fi

INIMG="$1"
OUTVID="$2"
FIXTIMEMS=$3
AFTERTIMEMS=$4
FPS=$5

if [ ! -f "$INIMG" ]; then
	echo "Input file $INIMG not found."
	exit 1
fi

# Make sure fix point frames exist
if [ ! -h "tmp/img0000.png" ]; then
	./mkfixframes.sh
else
	echo "Fix frames prepared."
fi

# Calculate frame indices
FIXFRAMES=$((FIXTIMEMS * FPS / 1000))
AFTERFRAMES=$((AFTERTIMEMS * FPS / 1000))
NUMFRAMES=$((FIXFRAMES + AFTERFRAMES + 1))
STIMFRAME=500
STARTFRAME=$((STIMFRAME - FIXFRAMES))
echo "Using sequence from $STARTFRAME to $((STARTFRAME+NUMFRAMES-1))."

# Put stimulus into sequence
IMGDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
rm -f $IMGDIR/tmp/img0500.png
echo ln -s $(readlink -f $INIMG) $IMGDIR/tmp/img0500.png
ln -s $(readlink -f $INIMG) $IMGDIR/tmp/img0500.png

# Delete previous
if [ -f "$OUTVID" ]; then
	echo "Removing previous output video $OUTVID."
	rm "$OUTVID"
fi

# Create video
echo ffmpeg -f image2 -framerate $FPS -start_number $STARTFRAME -i tmp/img%04d.png -c:v libvpx -b:v 650k -s 256x256 -r $FPS -vframes $NUMFRAMES "$OUTVID"
ffmpeg -f image2 -framerate $FPS -start_number $STARTFRAME -i tmp/img%04d.png -c:v libvpx -b:v 650k -s 256x256 -r $FPS -vframes $NUMFRAMES "$OUTVID"

