import typer
from pathlib import Path
import yaml
from omicverse.external.scTenifold import scTenifoldNet, scTenifoldKnk

app = typer.Typer()


@app.command(name="config")
def get_config_file(
        config_type: int = typer.Option(1, "--type", "-t",
                                        help="Type, 1: scTenifoldNet, 2: scTenifoldKnk",
                                        min=1, max=2),
        file_path: str = typer.Option(
                        ".config.yml",
                        "--path",
                        "-p",
                        help="Path to generate empty config file")) -> None:
    """Write an empty scTenifoldNet or scTenifoldKnk config YAML."""
    config = scTenifoldNet.get_empty_config() if config_type == 1 else scTenifoldKnk.get_empty_config()
    with open(Path(file_path), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


@app.command(name="net")
def build_net(config_file_path: str = typer.Option(...,
                                                   "--config",
                                                   "-c",
                                                   help="Loaded config file's path"),
              output_dir_path: str = typer.Option("./saved_net",
                                                  "--output",
                                                  "-o",
                                                  help="Output folder containing all analysis results"),
              ) -> None:
    """Build a scTenifoldNet from a config YAML and save the result."""
    with open(Path(config_file_path), "r") as f:
        data = yaml.safe_load(f)
    sc = scTenifoldNet.load_config(config=data)
    sc.build()
    sc.save(output_dir_path)


@app.command(name="knk")
def build_knk(config_file_path: str = typer.Option(...,
                                                   "--config",
                                                   "-c",
                                                   help="Loaded config file's path"),
              output_dir_path: str = typer.Option("./saved_knk",
                                                  "--output",
                                                  "-o",
                                                  help="Output folder containing all analysis results"),
              ) -> None:
    """Build a scTenifoldKnk from a config YAML and save the result."""
    with open(Path(config_file_path), "r") as f:
        data = yaml.safe_load(f)
    sc = scTenifoldKnk.load_config(config=data)
    sc.build()
    sc.save(output_dir_path)


if __name__ == '__main__':
    app()