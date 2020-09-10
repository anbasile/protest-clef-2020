import importlib
from pathlib import Path

import typer
from common import ModelType

app = typer.Typer()


@app.command()
def fit(
        model_type: ModelType,
        pretrained_model: str = typer.Argument(
            "google/mobilebert-uncased", help="The pretrained transformer to use"),
        dataset: Path = typer.Argument(...,
                                       help="The path of the folder with the training data"),
        crf_decoding: bool = typer.Option(False, help="Add crf decoding to model")):
    t = importlib.import_module('train', 'Trainer')
    """
        TODO
    """
    trainer = t.Trainer(
        model_type,
        pretrained_model,
        dataset,
        crf_decoding)
    trainer.run()
    typer.echo("Training model")


@app.command()
def predict():
    typer.echo("Running inference")


@app.command()
def evaluate():
    typer.echo("Running evaluation")


@app.command()
def serve():
    typer.echo("Running demo")


if __name__ == "__main__":
    app()
