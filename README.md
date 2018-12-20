#version1
Data after cleaning is stored in cleaned_data.csv
After feature engineering, the new dataset used for training is stored as new_data.csv
To implement training, simply run train.py, it will give the prediction accuracy.


#neural network method
Everything need to run this method is in the folder neural_network.
How to run:

1. Environment Settings

(1) Conda installation

wget https://repo.anaconda.com/archive/Anaconda2-5.3.1-Linux-x86_64.sh

sh Anaconda2-5.3.1-Linux-x86_64.sh

(2) Tensorflow installation

conda create -n tensorflow 

source activate tensorflow

conda install tensorflow

(3) Other packages installation

pip install keras

pip install nltk

Pip install gensim

pip install pandas

2. Pretrained Word2vec model download

https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

unzip it and put the .bin file in the neural_network/input folder

3. Run the program 

cd neural_network

Python LSTM.py

Note:  This will run a test on a small datset(train_sub.csv, test_sub.csv, test_sub_label.csv).

4.Compute the test accuracy

python  accu_compute.py test_sub_label.csv result_file.csv( you have to change the file name)
