__author__ = 'Per-Arne'


try:
    input = raw_input
except NameError:
    pass


def run_rnn_theano():
    pass

def run_chainer():
    import Chainer.Main

menu = {}
menu['1']="Start GRU-RNN (Theano)"
menu['2']="Start CHAR-RNN (Chainer)"
menu['3']="Exit"
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
        exit(0)

