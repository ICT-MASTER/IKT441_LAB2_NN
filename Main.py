__author__ = 'Per-Arne'
import Download
import time
import fnmatch
import os
import random
import uuid
import codecs
import Cleaner

try:
    input = raw_input
except NameError:
    pass


def run_rnn_theano():
    import Theano.Main

def run_chainer():
    import Chainer.Main


def download_articles():
    a_b_path = Download.download("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles.txt.0-9A-B.tar.gz", "./Download/")
    c_h_path = Download.download("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles.txt.C-H.tar.gz", "./Download/")
    i_n_path = Download.download("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles.txt.I-N.tar.gz", "./Download/")
    o_z_path = Download.download("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles.txt.O-Z.tar.gz", "./Download/")

def extract_articles():

    matches = []
    for root, dirnames, filenames in os.walk('./Download'):
        for filename in fnmatch.filter(filenames, '*.tar.gz'):
            matches.append(os.path.join(root, filename))

    for match in matches:
        Download.tar_gz(match)

def create_training_data():
    print("Loading articles... This may take a while")
    t_start = time.time()
    articles = []
    for root, dirnames, filenames in os.walk('./Articles'):
        for filename in fnmatch.filter(filenames, '*.txt'):
            articles.append(os.path.join(root, filename))
    print("Loading articles complete. Took {0} seconds...".format(time.time() - t_start))



    # Questions

    # Q1
    in_random_articles = input("Use random articles? [y/N]")
    if in_random_articles == "y":
        random.shuffle(articles)
        in_random_articles = True

    # Q2
    in_clean_file = input("Clean articles [Y/n]")
    if in_clean_file == "n":
        in_clean_file = False
    else:
        in_clean_file = True

    # Q3
    in_num_articles = input("Number or articles? [Default: 10]")
    try:
        num_articles = int(in_num_articles)
    except:
        num_articles = 10

    selected_articles = articles[0:min(len(articles), num_articles)-1]

    try:
        os.mkdir("./Training")
    except:
        pass

    training_filename = "Training-{0}-{1}-{2}-{3}.txt".format( \
        "Clean" if in_clean_file == True else "Dirty", \
        "Shuffle" if in_random_articles else "Iterate", \
        num_articles, \
        str(uuid.uuid4())[:8])
    for article in selected_articles:
        with codecs.open("./Training/" + training_filename, "a+", encoding="utf8") as file:
            with codecs.open(article,'r', encoding="utf8") as f:
                content = f.read()
                if in_clean_file == True:
                    content = Cleaner.clean(content)

                file.write(content)
    print("Created Training set named: {0}".format(training_filename))





menu = {}
menu['1']="Start GRU-RNN (Theano)"
menu['2']="Start CHAR-RNN (Chainer)"
menu['3']="Download articles"
menu['4']="Extract articles"
menu['5']="Create Training Data"
menu['6']="Exit"
while True:
    options = list(menu.keys())
    options.sort()
    for entry in options:
        print(entry, menu[entry])

    selection=input("Please Select:")
    if selection =='1':
        run_rnn_theano()
        break
    elif selection == '2':
        run_chainer()
        break
    elif selection == '3':
        download_articles()
    elif selection == '4':
        extract_articles()
    elif selection == '5':
        create_training_data()
    elif selection == '6':
        exit(0)

