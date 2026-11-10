"""Exercise the actual standalone script apps without root CLI registrations."""

from typer.testing import CliRunner

from pclass_dark.explore_ml_cli import app as explore_app
from pclass_dark.ml_model_cli import predict_app, train_app


def invoke_script(args, **options):
    """Route the script name to its real single-command Typer entry point."""
    apps = {"explore-ml": explore_app, "train-ml": train_app, "predict-ml": predict_app}
    return CliRunner().invoke(apps[args[0]], args[1:], **options)
