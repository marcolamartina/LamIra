#!/usr/bin/env bash 
find . -type f -name '*.ipynb' | while read i; do
    jupyter nbconvert --to script $i;
    i_no_ext=${i%.ipynb};
    mv "${i_no_ext}.txt" "${i_no_ext}.py"
done
