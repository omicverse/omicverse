from importlib import resources
from typing import Dict, List


predefined_signatures = dict(
    cell_cycle_human=resources.files("omicverse.datasets").joinpath("data_files/cell_cycle_human.gmt").__fspath__(),
    cell_cycle_mouse=resources.files("omicverse.datasets").joinpath("data_files/cell_cycle_mouse.gmt").__fspath__(),
    gender_human=resources.files("omicverse.datasets").joinpath("data_files/gender_human.gmt").__fspath__(),
    gender_mouse=resources.files("omicverse.datasets").joinpath("data_files/gender_mouse.gmt").__fspath__(),
    mitochondrial_genes_human=resources.files("omicverse.datasets").joinpath("data_files/mitochondrial_genes_human.gmt").__fspath__(),
    mitochondrial_genes_mouse=resources.files("omicverse.datasets").joinpath("data_files/mitochondrial_genes_mouse.gmt").__fspath__(),
    ribosomal_genes_human=resources.files("omicverse.datasets").joinpath("data_files/ribosomal_genes_human.gmt").__fspath__(),
    ribosomal_genes_mouse=resources.files("omicverse.datasets").joinpath("data_files/ribosomal_genes_mouse.gmt").__fspath__(),
    apoptosis_human=resources.files("omicverse.datasets").joinpath("data_files/apoptosis_human.gmt").__fspath__(),
    apoptosis_mouse=resources.files("omicverse.datasets").joinpath("data_files/apoptosis_mouse.gmt").__fspath__(),
    human_lung=resources.files("omicverse.datasets").joinpath("data_files/human_lung.gmt").__fspath__(),
    mouse_lung=resources.files("omicverse.datasets").joinpath("data_files/mouse_lung.gmt").__fspath__(),
    mouse_brain=resources.files("omicverse.datasets").joinpath("data_files/mouse_brain.gmt").__fspath__(),
    mouse_liver=resources.files("omicverse.datasets").joinpath("data_files/mouse_liver.gmt").__fspath__(),
    emt_human=resources.files("omicverse.datasets").joinpath("data_files/emt_human.gmt").__fspath__(),
    hallmark=resources.files("omicverse.datasets").joinpath("data_files/h.all.v7.5.1.symbols.gmt").__fspath__(),
    canonical_pathway=resources.files("omicverse.datasets").joinpath("data_files/c2.cp.v7.5.1.symbols.gmt").__fspath__(),
)


def load_signatures_from_file(input_file: str) -> Dict[str, List[str]]:
    signatures = {}
    with open(input_file) as fin:
        for line in fin:
            items = line.strip().split('\t')
            signatures[items[0]] = list(set(items[2:]))
    print(f"Loaded signatures from GMT file {input_file}.")
    return signatures
