from setuptools import setup

setup(
    name='little_questions',
    version='0.2',
    packages=['little_questions', 'little_questions.data',
              'little_questions.utils', 'little_questions.parsers'],
    package_data={'little_questions': ['res/en-us/*', 'models/*'],
                  'little_questions.data': ['*.txt']},
    include_package_data=True,
    install_requires=['numpy', 'scikit-learn', 'spacy', 'padaos', 'gensim'],
    url='https://github.com/JarbasAl/little_questions',
    license='MIT',
    author='jarbasAI',
    author_email='jarbasai@mailfence.com',
    description='question parser and classifier'
)
