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
    version='0.3.3',
    packages=['little_questions', 'little_questions.data',
              'little_questions.utils', 'little_questions.parsers',
              'little_questions.classifiers'],
    url='https://github.com/JarbasAl/little_questions',
    install_requires=['numpy', 'scikit-learn', 'spacy', 'padaos'],
    package_data={'': extra_files},
    include_package_data=True,
    license='MIT',
    author='jarbasAI',
    author_email='jarbasai@mailfence.com',
    description='question parser and classifier'
)
