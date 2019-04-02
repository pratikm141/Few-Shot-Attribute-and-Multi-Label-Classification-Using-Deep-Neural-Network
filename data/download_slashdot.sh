
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget

###### Download slashdot dataset

rm -rf slashdot
mkdir -p slashdot

cd slashdot
wget https://liquidtelecom.dl.sourceforge.net/project/meka/Datasets/SLASHDOT-F.arff


cd ..
python preprocess_slashdot.py
