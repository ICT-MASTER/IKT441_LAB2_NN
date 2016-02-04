import Download
import time
import random
import fnmatch
import os
import uuid
import Cleaner
import codecs

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu" # ,floatX=float32,allow_gc=False

#https://nsaunders.wordpress.com/2013/07/16/interestingly-the-sentence-adverbs-of-pubmed-central/
try:
    input = raw_input
except NameError:
    pass

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

    training_filename = "Training-{0}-{1}-{2}-{3}.txt".format(\
                                                      "Clean" if in_clean_file == True else "Dirty",\
                                                      "Shuffle" if in_random_articles else "Iterate",\
                                                      num_articles,\
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
menu['1']="Download articles"
menu['2']="Extract articles"
menu['3']="Create Training Data"
menu['4']="Train Neural Network"
menu['5']="Generate Sentences"
menu['6']="Exit"
while True:
    options = list(menu.keys())
    options.sort()
    for entry in options:
        print(entry, menu[entry])

    selection=input("Please Select:")
    if selection =='1':
        download_articles()
    elif selection == '2':
        extract_articles()
    elif selection == '3':
        create_training_data()
    elif selection == '4':


        #MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
        #INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/input.txt")

        learning_rate = input("Learning rate (default: {0}): ".format("0.001"))
        vocabulary_size = input("Vocabulary size (default: {0}): ".format("2000"))
        embedding_dim = input("Number of embedding dimensions (default: {0}): ".format(48))
        hidden_dim = input("Number of hidden dimensions (default: {0}): ".format(128))
        nepoch = input("Number of epochs (default: {0}): ".format(20))
        print_every = input("Print frequence (default: {0}): ".format(30))

        #################
        # Model list index
        try:
            os.mkdir("./Models")
        except: pass
        models = ["No Model"]
        for root, dirnames, filenames in os.walk('./Models'):
            for filename in fnmatch.filter(filenames, '*.dat.npz'):
                models.append(os.path.join(root, filename))

        print("--- Select Model File ---")
        for idx, value in enumerate(models):
            print("{0}. {1}".format(idx, value))

        ###############
        # Input for selecting model, if none or invalid was selected dont set model
        input_model_id = input("Select model: ")
        try:
            input_model_id = int(input_model_id)
            if input_model_id != 0:
                selected_model = models[input_model_id]
                os.environ["MODEL_OUTPUT_FILE"] = selected_model

                print("Selected {0} as model file".format(selected_model))
        except:
            pass

        #################
        # Input Data file (Training file)
        try:
            os.mkdir("./Training")
        except: pass
        training_files = []
        for root, dirnames, filenames in os.walk('./Training'):
            for filename in fnmatch.filter(filenames, '*.txt'):
                training_files.append(os.path.join(root, filename))

        # Break if no training files were found
        if len(training_files) == 0:
            print("No available training files, aborting...")
            break

        print("--- Select Training File ---")
        for idx, value in enumerate(training_files):
            print("{0}. {1}".format(idx, value))

        selected_training_file = None
        while selected_training_file is None:
            ###############
            # Input ofr selecting training
            input_training_id = input("Select Training file: ")
            try:
                input_training_id = int(input_training_id)
                selected_training_file = training_files[input_training_id]
            except:
                print("Invalid choice...")


        print("--- Select RNN Model ---")
        rnn_models = ["RNNTheano", "GRUTheano"]
        for idx, value in enumerate(rnn_models):
            print("{0}. {1}".format(idx, value))

        try:
            in_rnn_model = int(input("Select rnn model (Default: 0): "))
            in_rnn_model = rnn_models[in_rnn_model]
        except:
            in_rnn_model = "GRUTheano"
        os.environ["RNN_MODEL"] = in_rnn_model


        os.environ["LEARNING_RATE"] = learning_rate or "0.001"
        os.environ["VOCABULARY_SIZE"] = vocabulary_size or "2000"
        os.environ["EMBEDDING_DIM"] = embedding_dim or "48"
        os.environ["HIDDEN_DIM"] = hidden_dim or "128"
        os.environ["NEPOCH"] = nepoch or "20"
        os.environ["PRINT_EVERY"] = print_every or "30"

        os.environ["INPUT_DATA_FILE"] = selected_training_file

        import rnn_train


    elif selection == '5':
        import rnn_utils



        #################
        # Model list index
        try:
            os.mkdir("./Models")
        except: pass
        models = ["No Model"]
        for root, dirnames, filenames in os.walk('./'):
            for filename in fnmatch.filter(filenames, '*.dat.npz'):
                models.append(os.path.join(root, filename))

        print("--- Select Model File ---")
        for idx, value in enumerate(models):
            print("{0}. {1}".format(idx, value))

        ###############
        # Input for selecting model, if none or invalid was selected dont set model
        input_model_id = input("Select model: ")
        try:
            input_model_id = int(input_model_id)
            if input_model_id != 0:
                model_path = models[input_model_id]

                print("Selected {0} as model file".format(model_path))
        except:
            pass

        #################
        # Input Data file (Training file)
        try:
            os.mkdir("./Training")
        except: pass
        training_files = []
        for root, dirnames, filenames in os.walk('./Training'):
            for filename in fnmatch.filter(filenames, '*.txt'):
                training_files.append(os.path.join(root, filename))

        # Break if no training files were found
        if len(training_files) == 0:
            print("No available training files, aborting...")
            break

        print("--- Select Training File ---")
        for idx, value in enumerate(training_files):
            print("{0}. {1}".format(idx, value))

        selected_training_file = None
        while selected_training_file is None:
            ###############
            # Input ofr selecting training
            input_training_id = input("Select Training file: ")
            try:
                input_training_id = int(input_training_id)
                selected_training_file = training_files[input_training_id]
            except:
                print("Invalid choice...")

        num_sentences = input("How many sentences to generate?: ")
        try:
            num_sentences = int(num_sentences)
        except:
            print("Invalid number!")
            break




        model_name = os.path.splitext(model_path)[0]


        x_train, y_train, word_to_index, index_to_word = rnn_utils.load_data(selected_training_file)
        model = rnn_utils.load_model_parameters_theano(model_path)
        rnn_utils.generate_sentences(model, num_sentences, index_to_word, word_to_index, model_name)



    elif selection == '6':
        exit(0)
    else:
        print("Unknown Option Selected!")



"""
out_file = open("inpux.txt", "w")
i = 0
num = 50
for (dir, _, files) in os.walk("./Articles"):
    for f in files:
        path = os.path.join(dir, f)
        if os.path.exists(path):
            with open(path, "r") as file:
                content = file.read()
                cleaned_content = Cleaner.clean(content)
                out_file.write(cleaned_content)

    print("{0}/{1}".format(i, num))

    if i == num:
        break
    i += 1

out_file.close()
"""


# Select a article
"""select_article = 5
i = 0
selected_article = None
for (dir, _, files) in os.walk("./Articles"):
    for f in files:
        path = os.path.join(dir, f)
        if os.path.exists(path):

            if i == select_article:
                selected_article = path
                break

            i += 1
"""
