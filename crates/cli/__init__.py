import sys

import click

from .commands import (
    config_cmd,
    data_cmd,
    list_cmd,
    train_cmd,
)


@click.group()
@click.pass_context
def app(ctx):
    for stream in (sys.stdout, sys.stderr):
        if stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]
    ctx.ensure_object(dict)


app.add_command(train_cmd.train)
app.add_command(train_cmd.from_config)
app.add_command(train_cmd.resume)
app.add_command(data_cmd.app)
app.add_command(list_cmd.app)
app.add_command(config_cmd.app)


if __name__ == "__main__":
    app()
