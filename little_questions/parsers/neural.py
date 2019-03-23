try:
    from padatious import IntentContainer
except ImportError:
    print("padatious not found, run")
    print("pip install fann2==1.0.7")
    print("pip install padatious==0.4.5")
    raise

from little_questions.parsers import BasicQuestionParser


class NeuralQuestionParser(BasicQuestionParser):

    def __init__(self, lang="en-us"):
        self.container = IntentContainer('intent_cache')
        self._intents = []
        self.lang = lang
        self.register_default_intents()
        self.container.train()

    def parse(self, utterance):
        data = {"Question": utterance}
        match = self.container.calc_intent(utterance)
        data["QuestionIntent"] = match.name
        data.update(match.matches)
        data["conf"] = match.conf
        return data


if __name__ == "__main__":
    from pprint import pprint
    from little_questions.data import SAMPLE_QUESTIONS
    import random

    parser = NeuralQuestionParser()
    b_parser = BasicQuestionParser()

    questions = SAMPLE_QUESTIONS
    random.shuffle(questions)

    for q in questions:
        data = parser.parse(q)
        datab = b_parser.parse(q)
        if data["QuestionIntent"] != datab["QuestionIntent"]:
            print("Q:", q)
            print("Intent:", datab['QuestionIntent'])
            datab.pop("QuestionIntent")
            datab.pop("Question")
            print("Intent Data:", datab)
            print("Neural Intent:", data['QuestionIntent'])
            data.pop("QuestionIntent")
            data.pop("Question")
            print("Neural Data:", data)
            print("___")

    """Q: For how long is an elephant pregnant ?

        Intent: unknown
        Neural Intent: select_option
        ___
        Q: How cold should a refrigerator be ?
        
        Intent: unknown
        Neural Intent: relate_time_and_attribute
        ___
        Q: Why in tennis are zero points called love ?
        
        Intent: unknown
        Neural Intent: explanation
        ___
        Q: What planet is known as the `` red '' planet ?
        
        Intent: describe_attribute
        Neural Intent: select_option
        ___
        Q: In Poland , where do most people live ?
        
        Intent: unknown
        Neural Intent: place
        ___
        """