from setuptools import setup

setup(
    name='little_questions',
    version='0.7.0',
    packages=['little_questions',
              'little_questions.classifiers',
              'little_questions.features'],
    url='https://github.com/OpenJarbas/little_questions',
    install_requires=['numpy==1.16.1', 'scikit-learn', "nltk"],
    include_package_data=True,
    license='Apache2.0',
    author='jarbasAI',
    author_email='jarbasai@mailfence.com',
    description='question parser and classifier'
)
