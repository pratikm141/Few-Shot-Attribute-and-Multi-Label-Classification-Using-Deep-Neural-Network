
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget
checkCmd unzip

###### Download emotions dataset

rm -rf medical
mkdir -p medical

cd medical
wget https://sci2s.ugr.es/keel/dataset/data/multilabel/medical.zip


unzip medical.zip

cd ..
python preprocess_medical.py

