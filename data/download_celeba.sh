
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget
checkCmd unzip

###### Download birds dataset

rm -rf celeba
mkdir -p celeba

cd celeba
wget -O celeba_resized.zip https://ndownloader.figshare.com/files/14759408

wget -O list_attr_celeba.txt https://ndownloader.figshare.com/files/14759465

unzip celeba_resized.zip

cd ..
python preprocess_celeba.py

