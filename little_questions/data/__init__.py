from little_questions.settings import DATA_PATH
from os.path import join
import random


with open(join(DATA_PATH, "questions.txt")) as f:
    SAMPLE_QUESTIONS = f.readlines()


def random_question():
    return random.choice(SAMPLE_QUESTIONS)


if __name__ == "__main__":
    for i in range(0, 20):
        print(random_question())