__author__ = 'Per-Arne'
import os
import ConfigParser

try:
    input = raw_input
except NameError:
    pass


def select_menu(path, extension="*", title="File's Found"):
    import fnmatch

    # Find all files
    files = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, extension):
            files.append(os.path.join(root, filename))

    # Display all files as a list
    print("--- {0} ---".format(title))
    for idx, file in enumerate(files):
        print("{0}. {1}".format(idx, file))

    selected_file = None
    while selected_file == None:

        try:
            selected_file = files[int(input("Select File: "))]
        except:
            pass

    return selected_file








menu = {}
menu['1']="Train"
menu['2']="Sample"
menu['3']="Exit"
while True:
    options = list(menu.keys())
    options.sort()
    print("--- CHAR-RNN with Chainer ---")
    for entry in options:
        print("{0}. {1}".format(entry, menu[entry]))

    selection=input("Please Select:")



    if selection =='1':
        
        section_name = "Training"
        config_ini_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Config.ini")
        config = ConfigParser.ConfigParser()
        config.read(config_ini_path)

        # Edit config
        if input("Edit Config? (y/N): ") == "y":
            options = config.options(section_name)
            print("--- Training Parameters --- ")
            for idx, option in enumerate(options):
                opt_val = config.get(section_name, option)
                if input("Change {0} (Current: {1}) [y/N]: ".format(option, opt_val)) == "y":
                    new_val = input("New Value: ")
                    config.set(section_name, option, new_val)
                    print("{0} = {1}".format(option, new_val))

        # Update config
        with open(config_ini_path, 'wb') as configfile:
            config.write(configfile)


        # Select Training file
        training_files_path = "./Training/"
        selected_file_path = select_menu(training_files_path, "*.txt", "Training Files")
        print("Selected {0}..".format(selected_file_path))

        os.environ["RNN_TRAINING_FILE"] =               selected_file_path
        os.environ["RNN_DATA_DIR"] =                    config.get(section_name, "data_dir")
        os.environ["RNN_CHECKPOINT_DIR"] =              config.get(section_name, "checkpoint_dir")
        os.environ["RNN_GPU"] =                         config.get(section_name, "gpu")
        os.environ["RNN_RNN_SIZE"] =                    config.get(section_name, "rnn_size")
        os.environ["RNN_LEARNING_RATE"] =               config.get(section_name, "learning_rate")
        os.environ["RNN_LEARNING_RATE_DECAY"] =         config.get(section_name, "learning_rate_decay")
        os.environ["RNN_LEARNING_RATE_DECAY_AFTER"] =   config.get(section_name, "learning_rate_decay_after")
        os.environ["RNN_DECAY_RATE"] =                  config.get(section_name, "decay_rate")
        os.environ["RNN_DROPOUT"] =                     config.get(section_name, "dropout")
        os.environ["RNN_SEQ_LENGTH"] =                  config.get(section_name, "seq_length")
        os.environ["RNN_BATCHSIZE"] =                   config.get(section_name, "batchsize")
        os.environ["RNN_EPOCHS"] =                      config.get(section_name, "epochs")
        os.environ["RNN_GRAD_CLIP"] =                   config.get(section_name, "grad_clip")
        os.environ["RNN_INIT_FROM"] =                   config.get(section_name, "init_from")

        import train

    elif selection == '2':
        section_name = "Sampling"
        config_ini_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Config.ini")
        config = ConfigParser.ConfigParser()
        config.read(config_ini_path)

        # Edit config
        if input("Edit Config? (y/N): ") == "y":
            options = config.options(section_name)
            print("--- Sample Parameters --- ")
            for idx, option in enumerate(options):
                opt_val = config.get(section_name, option)
                if input("Change {0} (Current: {1}) [y/N]: ".format(option, opt_val)) == "y":
                    new_val = input("New Value: ")
                    config.set(section_name, option, new_val)
                    print("{0} = {1}".format(option, new_val))

        # Update config
        with open(config_ini_path, 'wb') as configfile:
            config.write(configfile)

        os.environ["RNN_MODEL"] = config.get(section_name, "model")
        os.environ["RNN_VOCABULARY"] = config.get(section_name, "vocabulary")
        os.environ["RNN_SEED"] = config.get(section_name, "seed")
        os.environ["RNN_SAMPLE"] = config.get(section_name, "sample")
        os.environ["RNN_PRIMETEXT"] = config.get(section_name, "primetext")
        os.environ["RNN_LENGTH"] = config.get(section_name, "length")
        os.environ["RNN_GPU"] = config.get(section_name, "gpu")

        import sample
        sample.run()
    elif selection == '3':
        exit(0)


