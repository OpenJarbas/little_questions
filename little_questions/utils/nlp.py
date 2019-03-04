SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod",
              "ccomp", "complm",
              "hmod", "infmod", "xcomp", "rcmod", "poss", " possessive"]
COMPOUNDS = ["compound"]
PREPOSITIONS = ["prep"]


def get_subs_from_conjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if
                             tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(get_subs_from_conjunctions(moreSubs))
    return moreSubs


def get_objs_from_conjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if
                             tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(get_objs_from_conjunctions(moreObjs))
    return moreObjs


def get_verbs_from_conjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend(
                [tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(get_verbs_from_conjunctions(moreVerbs))
    return moreVerbs


def find_subs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = is_negated(head)
            subs.extend(get_subs_from_conjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return find_subs(head)
    elif head.pos_ == "NOUN":
        return [head], is_negated(tok)
    return [], False


def is_negated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False


def find_svs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = get_all_subs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append(
                    (sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


def get_objs_from_prepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend([tok for tok in dep.rights if tok.dep_ in OBJECTS or (
                    tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs


def get_adjectives(toks):
    toks_with_adjectives = []
    for tok in toks:
        adjs = [left for left in tok.lefts if left.dep_ in ADJECTIVES]
        adjs.append(tok)
        adjs.extend([right for right in tok.rights if tok.dep_ in ADJECTIVES])
        tok_with_adj = " ".join([adj.lower_ for adj in adjs])
        toks_with_adjectives.extend(adjs)

    return toks_with_adjectives


def get_objs_from_attrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(get_objs_from_prepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None


def get_obj_from_xcomp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(get_objs_from_prepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None


def get_all_subs(v):
    verbNegated = is_negated(v)
    subs = [tok for tok in v.lefts if
            tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(get_subs_from_conjunctions(subs))
    else:
        foundSubs, verbNegated = find_subs(v)
        subs.extend(foundSubs)
    return subs, verbNegated


def get_all_objs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(get_objs_from_prepositions(rights))

    potentialNewVerb, potentialNewObjs = get_obj_from_xcomp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(
            potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(get_objs_from_conjunctions(objs))
    return v, objs


def get_all_objs_with_adjectives(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]

    if len(objs) == 0:
        objs = [tok for tok in rights if tok.dep_ in ADJECTIVES]

    objs.extend(get_objs_from_prepositions(rights))

    potentialNewVerb, potentialNewObjs = get_obj_from_xcomp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(
            potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(get_objs_from_conjunctions(objs))
    return v, objs


def find_svos(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, verbNegated = get_all_subs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = get_all_objs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = is_negated(obj)
                    svos.append((sub.lower_,
                                 "!" + v.lower_ if verbNegated or objNegated else v.lower_,
                                 obj.lower_))
    return svos


def find_svaos(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, verbNegated = get_all_subs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = get_all_objs_with_adjectives(v)
            for sub in subs:
                for obj in objs:
                    objNegated = is_negated(obj)
                    obj_desc_tokens = generate_left_right_adjectives(obj)
                    sub_compound = generate_sub_compound(sub)
                    svos.append((" ".join(tok.lower_ for tok in sub_compound),
                                 "!" + v.lower_ if verbNegated or objNegated else v.lower_,
                                 " ".join(
                                     tok.lower_ for tok in obj_desc_tokens)))
    return svos


def generate_sub_compound(sub):
    sub_compunds = []
    for tok in sub.lefts:
        if tok.dep_ in COMPOUNDS:
            sub_compunds.extend(generate_sub_compound(tok))
    sub_compunds.append(sub)
    for tok in sub.rights:
        if tok.dep_ in COMPOUNDS:
            sub_compunds.extend(generate_sub_compound(tok))
    return sub_compunds


def generate_left_right_adjectives(obj):
    obj_desc_tokens = []
    for tok in obj.lefts:
        if tok.dep_ in ADJECTIVES:
            obj_desc_tokens.extend(generate_left_right_adjectives(tok))
    obj_desc_tokens.append(obj)

    for tok in obj.rights:
        if tok.dep_ in ADJECTIVES:
            obj_desc_tokens.extend(generate_left_right_adjectives(tok))

    return obj_desc_tokens


if __name__ == "__main__":
    from little_questions.data import SAMPLE_QUESTIONS
    from little_questions.settings import nlp
    import random
    from pprint import pprint

    questions = SAMPLE_QUESTIONS
    random.shuffle(questions)

    for q in questions:
        parse = nlp(q)
        print("Q:", q)
        pprint(find_svs(parse))
        print("____")