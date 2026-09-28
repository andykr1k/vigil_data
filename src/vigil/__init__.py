"""Vigil: RealSense leg pose estimation with a live 3D dashboard."""


def main() -> None:
    from .cli import main as cli_main

    cli_main()
