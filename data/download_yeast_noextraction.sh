
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget

###### Download yeast dataset

rm -rf yeast
mkdir -p yeast

cd yeast
wget https://excellmedia.dl.sourceforge.net/project/mulan/datasets/yeast.rar

cd ..

