import Download
import os
import RNN.Cleaner as Cleaner

#https://nsaunders.wordpress.com/2013/07/16/interestingly-the-sentence-adverbs-of-pubmed-central/

# Download Packs
a_b_path = Download.download("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles.txt.0-9A-B.tar.gz", "./Download/")
c_h_path = Download.download("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles.txt.C-H.tar.gz", "./Download/")
i_n_path = Download.download("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles.txt.I-N.tar.gz", "./Download/")
o_z_path = Download.download("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles.txt.O-Z.tar.gz", "./Download/")

#a_b_txt_path = Download.tar_gz(a_b_path)
#c_h_txt_path = Download.tar_gz(c_h_path)
#i_n_txt_path = Download.tar_gz(i_n_path)
#o_z_txt_path = Download.tar_gz(o_z_path)





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

out_file = open("inpux.txt", "w")
i = 0
num = 10
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
import RNN.NeuralNetwork
