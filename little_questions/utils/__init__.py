import requests


def replace_coreferences(text, nlp=None):
    # "My sister has a dog. She loves him." -> "My sister has a dog. My sister loves a dog."

    # """
    # London is the capital and most populous city of England and  the United Kingdom.
    # Standing on the River Thames in the south east of the island of Great Britain,
    # London has been a major settlement  for two millennia.  It was founded by the Romans,
    # who named it Londinium.
    # """ -> """
    # London is the capital and most populous city of England and  the United Kingdom.
    # Standing on the River Thames in the south east of the island of Great Britain,
    # London has been a major settlement  for two millennia.  London was founded by the Romans,
    # the Romans named London Londinium.
    # """
    try:
        if nlp is not None:
            doc = nlp(text)
            text = doc._.coref_resolved
        else:
            # neural coref catches "It" but fails for the "who" in romans
            # it also fails on long texts ocasionally
            text = neuralcoref_demo(text)
            # cogcomp catches some more stuff
            text = cogcomp_coref_resolution(text)
    except:
        pass
    return text


def neuralcoref_demo(text):
    try:
        params = {"text": text}
        r = requests.get("https://coref.huggingface.co/coref",
                         params=params).json()
        text = r["corefResText"] or text
    except Exception as e:
        print(e)
    return text


def cogcomp_coref_resolution(text):
    # use the source https://cogcomp.org/page/demo_view/Coref
    replaces = ["he", "she", "it", "they", "them", "these", "whom", "whose",
                "who", "its", "it's"]
    url = "https://cogcomp.org/demo_files/Coref.php"
    data = {"lang": "en", "text": text}
    data = requests.post(url, json=data).json()
    links = data["links"]
    node_ids = {}
    replace_map = {}
    for n in data["nodes"]:
        node_ids[int(n["id"])] = n["name"]
    for l in links:
        # only replace some stuff
        if node_ids[l["target"]].lower() not in replaces:
            continue
        replace_map[node_ids[l["target"]]] = node_ids[l["source"]]
    for r in replace_map:
        text = text.replace(r, replace_map[r])
    return text


def normalize(text, remove_articles=True, solve_corefs=False, nlp=None):
    """ English string normalization """
    text = str(text)
    words = text.split()  # this also removed extra spaces
    normalized = ""

    for word in words:
        if remove_articles and word in ["the", "a", "an"]:
            continue

        # Expand common contractions, e.g. "isn't" -> "is not"
        contraction = ["ain't", "aren't", "can't", "could've", "couldn't",
                       "didn't", "doesn't", "don't", "gonna", "gotta",
                       "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's",
                       "how'd", "how'll", "how's", "I'd", "I'll", "I'm",
                       "I've", "isn't", "it'd", "it'll", "it's", "mightn't",
                       "might've", "mustn't", "must've", "needn't",
                       "oughtn't",
                       "shan't", "she'd", "she'll", "she's", "shouldn't",
                       "should've", "somebody's", "someone'd", "someone'll",
                       "someone's", "that'll", "that's", "that'd", "there'd",
                       "there're", "there's", "they'd", "they'll", "they're",
                       "they've", "wasn't", "we'd", "we'll", "we're", "we've",
                       "weren't", "what'd", "what'll", "what're", "what's",
                       "whats",  # technically incorrect but some STT outputs
                       "what've", "when's", "when'd", "where'd", "where's",
                       "where've", "who'd", "who'd've", "who'll", "who're",
                       "who's", "who've", "why'd", "why're", "why's", "won't",
                       "won't've", "would've", "wouldn't", "wouldn't've",
                       "y'all", "ya'll", "you'd", "you'd've", "you'll",
                       "y'aint", "y'ain't", "you're", "you've"]
        if word in contraction:
            expansion = ["is not", "are not", "can not", "could have",
                         "could not", "did not", "does not", "do not",
                         "going to", "got to", "had not", "has not",
                         "have not", "he would", "he will", "he is",
                         "how did",
                         "how will", "how is", "I would", "I will", "I am",
                         "I have", "is not", "it would", "it will", "it is",
                         "might not", "might have", "must not", "must have",
                         "need not", "ought not", "shall not", "she would",
                         "she will", "she is", "should not", "should have",
                         "somebody is", "someone would", "someone will",
                         "someone is", "that will", "that is", "that would",
                         "there would", "there are", "there is", "they would",
                         "they will", "they are", "they have", "was not",
                         "we would", "we will", "we are", "we have",
                         "were not", "what did", "what will", "what are",
                         "what is",
                         "what is", "what have", "when is", "when did",
                         "where did", "where is", "where have", "who would",
                         "who would have", "who will", "who are", "who is",
                         "who have", "why did", "why are", "why is",
                         "will not", "will not have", "would have",
                         "would not", "would not have", "you all", "you all",
                         "you would", "you would have", "you will",
                         "you are not", "you are not", "you are", "you have"]
            word = expansion[contraction.index(word)]

        normalized += " " + word

    if solve_corefs:
        normalized = replace_coreferences(normalized[1:], nlp)

    return normalized.strip()
