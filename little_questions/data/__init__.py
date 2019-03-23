from little_questions.settings import DATA_PATH
from os.path import join
import random

SAMPLE_QUESTIONS = []

with open(join(DATA_PATH, "questions.txt")) as f:
    for q in f.readlines():
        SAMPLE_QUESTIONS.append(" ".join(q.split(" ")[1:]))
with open(join(DATA_PATH, "questions_test.txt")) as f:
    for q in f.readlines():
        SAMPLE_QUESTIONS.append(" ".join(q.split(" ")[1:]))


def random_question():
    return random.choice(SAMPLE_QUESTIONS)


if __name__ == "__main__":
    for i in range(0, 20):
        print(random_question())
