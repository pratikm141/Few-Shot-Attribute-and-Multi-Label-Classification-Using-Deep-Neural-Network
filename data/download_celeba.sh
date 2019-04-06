
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget
checkCmd unzip

###### Download celeba dataset

rm -rf celeba
mkdir -p celeba

cd celeba
wget -O list_attr_celeba.txt https://ndownloader.figshare.com/files/14759465

mkdir celeba_img
cd celeba_img
wget -O resized_celebA.zip https://ndownloader.figshare.com/files/14759408

unzip resized_celebA.zip

cd ..

cd ..


python preprocess_celeba.py
