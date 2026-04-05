#!/bin/bash
 
OUTPUT="dump.txt"
> "$OUTPUT"
 
for file in *.py; do
    echo "================ $file ================" >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
    echo -e "\n\n" >> "$OUTPUT"
done
 
echo "Done → $OUTPUT"