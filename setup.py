from distutils.core import setup

setup(
    name='protesta',
    version='0.0.1',
    description='Detect protest events in news articles',
    author='Angelo Basile',
    author_email='me@angelobasile.it',
    install_requires=[
        "nlp>=0.4",
        "numpy<1.19",
        "tensorflow-addons",
        "tensorflow==2.3.1",
        "transformers==3.5",
        "typer>=0.3.1",
    ],
    entry_points={"console_scripts": ["protesta=protesta:app"]})
