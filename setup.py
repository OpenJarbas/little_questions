from setuptools import setup
import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


extra_files = package_files('little_questions')

setup(
    name='little_questions',
    version='0.5.1',
    packages=['little_questions', 'little_questions.data',
              'little_questions.utils', 'little_questions.parsers',
              'little_questions.classifiers'],
    url='https://github.com/JarbasAl/little_questions',
    install_requires=['numpy', 'scikit-learn', 'gensim', 'padaos', "nltk",
                      'simple_NER>=0.1.10', "text_classifikation",
                      "fann2==1.0.7", "padatious>=0.4.5"],
    package_data={'': extra_files},
    include_package_data=True,
    license='MIT',
    author='jarbasAI',
    author_email='jarbasai@mailfence.com',
    description='question parser and classifier'
)
