import importlib
from pathlib import Path

import typer
from common import ModelType, EncodingMode

app = typer.Typer()


@app.command()
def fit(
    model_type: ModelType = typer.Option(
        ModelType.SequenceTagger, help="The type of model to train"),
    pretrained_model: str = typer.Argument(...,
                                           help="The pretrained transformer to use"),
    dataset: Path = typer.Argument(...,
                                   help="The path of the folder with the training data"),
    crf_decoding: bool = typer.Option(
        False, help="Add crf decoding to model"),
    encoding: EncodingMode = typer.Option(
        EncodingMode.DocumentWise, help="Encode a document as a whole or sentence by sentence"),
    data_size: float = typer.Option(
        1.0, max=1.0, min=0.1, help='Percentage of training data to use to fit the model')
):
    t = importlib.import_module('train', 'Trainer')
    """
        TODO
    """
    trainer = t.Trainer(
        model_type,
        pretrained_model,
        dataset,
        crf_decoding,
        encoding,
        data_size)
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
