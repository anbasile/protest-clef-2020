from distutils.core import setup

setup(
    name='protesta',
    version='0.0.1',
    description='Detect protest events in news articles',
    author='Angelo Basile',
    author_email='me@angelobasile.it',
    install_requires=[
        "nlp>=0.4",
        "tensorflow-addons",
        "tensorflow==2.4.0",
        "transformers==4.3.3",
        "typer>=0.3.1",
    ],
    entry_points={"console_scripts": ["protesta=protesta:app"]})
