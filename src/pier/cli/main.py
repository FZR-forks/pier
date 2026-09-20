import typer
from typer import Typer

from pier import __version__
from pier.cli.analyze import analyze_command, check_command
from pier.cli.critique import critique_app
from pier.cli.jobs import jobs_app, start
from pier.cli.view import view_command


def version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit()


app = Typer(no_args_is_help=True)


@app.callback()
def main(
    version: bool | None = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
) -> None:
    pass


app.add_typer(jobs_app, name="job", help="Manage jobs.")
app.add_typer(critique_app, name="critique", help="Run sandboxed critiques.")
app.command(name="check", help="Check task quality against a rubric.")(check_command)
app.command(name="analyze", help="Analyze trial trajectories.")(analyze_command)
app.command(name="run", help="Start a job.")(start)
app.command(name="view", help="Start web server to browse trajectory files.")(
    view_command
)


if __name__ == "__main__":
    app()
