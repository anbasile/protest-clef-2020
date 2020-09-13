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
    typer.echo("Training model")
    trainer.run()


@app.command()
def predict(
    model_dir: Path,
    input_file: Path):
    typer.echo("Running inference")
    i = importlib.import_module('inference', 'Inferencer')
    """
        TODO
    """
    inference = i.Inferencer(model_dir, input_file)
    inference.run()




@app.command()
def evaluate():
    typer.echo("TODO Running evaluation")
    raise NotImplementedError


@app.command()
def serve():
    typer.echo("TODO Running demo")
    raise NotImplementedError


if __name__ == "__main__":
    app()
