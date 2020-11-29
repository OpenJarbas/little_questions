from setuptools import setup

setup(
    name='little_questions',
    version='0.6.0',
    packages=['little_questions', 'little_questions.data',
              'little_questions.utils', 'little_questions.parsers',
              'little_questions.classifiers', 'little_questions.features'],
    url='https://github.com/OpenJarbas/little_questions',
    install_requires=['numpy==1.16.1', 'scikit-learn', 'gensim', 'padaos',
                      "nltk", 'simple_NER>=0.6.1', "fann2==1.0.7",
                      "padatious>=0.4.5"],
    include_package_data=True,
    license='Apache2.0',
    author='jarbasAI',
    author_email='jarbasai@mailfence.com',
    description='question parser and classifier'
)
