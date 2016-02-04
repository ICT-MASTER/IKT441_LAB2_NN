


def clean(content):

    content = get_body(content)
    content = strip_few_words_on_line(content)

    return content


def get_body(content):
    content = content.split('==== Body', 1)[-1]
    content = content.rsplit("==== Refs", 1)[0]
    return content

def strip_few_words_on_line(content):
    new_content = ""
    for line in content.split("\n"):

        # Count number of letter words in line
        words  = sum(c.isalpha() for c in line)

        split = line.split(" ")

        # Require more than 5 words on a line
        if len(split) > 5 and words > 5:
            new_content += line + "\n"

    return new_content




