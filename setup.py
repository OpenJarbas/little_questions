from setuptools import setup

setup(
    name='little_questions',
    version='0.1',
    packages=['little_questions', 'little_questions.data',
              'little_questions.utils', 'little_questions.parsers',
              'little_questions.classifiers.logreg',
              'little_questions.classifiers.gradboost'],
    url='',
    license='MIT',
    author='jarbasAI',
    author_email='jarbasai@mailfence.com',
    description='question parser and classifier'
)
