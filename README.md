# Few-Shot-Attribute-and-Multi-Label-Classification-Using-Deep-Neural-Network
Code for Few-Shot Attribute and Multi-Label Classification Using Deep Neural Network implemented in pytorch

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
  * Note: We currently only support Python 3+.
- Install unzip and unrar
- Then download the dataset by following the [instructions](#Datasets) below.
- Follow the [run instructions](#Experiments) below for training and testing the model.

## Datasets
- Go to the download folder
- Run "sh download_&lt;dataset&gt;.sh" to download and preprocess the datasets
- The prepocessing code for each dataset will be automatically called by the corresponding download_&lt;dataset&gt;.sh file
- Some of the datasets are downloaded directly from the main repository.
- Some of the datasets which need some modifications are uploaded to figshare after modification. 
- Following datasets are used:
  * LFW - images https://ndownloader.figshare.com/files/14758619, attributes http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
  * CelebA - images https://ndownloader.figshare.com/files/14759408, attributes https://ndownloader.figshare.com/files/14759465
  * yeast - https://excellmedia.dl.sourceforge.net/project/mulan/datasets/yeast.rar
  * emotions - https://sci2s.ugr.es/keel/dataset/data/multilabel/emotions.zip
  * birds - http://sourceforge.net/projects/mulan/files/datasets/birds.rar
  * enron - https://sci2s.ugr.es/keel/dataset/data/multilabel/enron.zip
  * medical - https://sci2s.ugr.es/keel/dataset/data/multilabel/medical.zip
  * llog - https://liquidtelecom.dl.sourceforge.net/project/meka/Datasets/LLOG-F.arff
  * slashdot - https://liquidtelecom.dl.sourceforge.net/project/meka/Datasets/SLASHDOT-F.arff
  * scene - https://sci2s.ugr.es/keel/dataset/data/multilabel/scene.zip

## Experiments
- Run the &lt;...&gt;_train_test.py file to get the results for our proposed Multi-Label model
- Run the &lt;...&gt;_train_test_baseline.py file to get the results for the baseline
- Provide additional choices:
  * --gpu _ :gpu id on which the program will be run
  * --epochs _ : Number of epochs(default 20) for Phase 1 (train on first half of labels and test on remaining) and Phase 2 (train on second half of labels and test on remaining). Each epoch has 1000 episodes
  * --shot _ : N-shot - Number of support examples per label (deafult 20).
- Average Test F1 Score will be displayed after the Phase 1 and Phase 2 tests are completed
