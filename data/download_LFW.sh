
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget
checkCmd unzip

###### Download dataset

rm -rf lfw
mkdir -p lfw

cd lfw
wget -O lfw-deepfunneled_reduced_bkup.zip https://ndownloader.figshare.com/files/14758619

wget -O lfw_attributes.txt http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt

unzip lfw-deepfunneled_reduced.zip

cd ..
python preprocess_LFW.py
