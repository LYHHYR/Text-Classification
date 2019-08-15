from pandas import read_csv

# load raw data
def loader(file, path):
    """ load raw feature from file ("train" or "test")"""
    raw_data = read_csv(path + '/' + file, header=0, delimiter='\t')
    return (raw_data)

######################################
#      Testing functions below       #
######################################
def test_loader():
    path = '~/Desktop/NLP-Beginner/task1'
    raw_train = loader('train.tsv', path)
    print(raw_train)

if __name__ == "__main__":
    test_loader()