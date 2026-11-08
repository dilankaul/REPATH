# Third-Party Notices

This repository includes Python implementations of five published web element relocalization methods for comparative evaluation. The original repositories and reference revisions are listed below.

## Baseline Methods

| Method | Original Repository | Reference Commit | Upstream License |
|---|---|---|---|
| WATER | [shauvik/water](https://github.com/shauvik/water) | `c57311e56afb8e52ccd315e460e1a6f5084a7b95` | Not identified |
| VISTA | [saltlab/vista](https://github.com/saltlab/vista) | `7bf2a588897637e0e1c9a2cc33e4d372a62bcaf0` | Apache-2.0 |
| SFTM | [amaris/sftm_tree_matching](https://github.com/amaris/sftm_tree_matching) | `4f8683b0e5a21050d1679f1c350f558a5a4171e5` | GPL-3.0-or-later |
| WebEvo | [webevoexp/webevo](https://github.com/webevoexp/webevo) | `d195630b8dd5d85790d5aebb5c81c29bd03f01ca` | Not identified |
| Similo | [michelnass/Similo2](https://github.com/michelnass/Similo2) | `743ddede7df92814ddc3fb7f96e09bdb508e285a` | Not identified |

## Licensing and Attribution

**SFTM:** The Python implementation is a port of the original SFTM implementation, Copyright (C) 2021 Mantu, Sacha Brisset, licensed under GPL-3.0-or-later. A copy of GNU GPL version 3 is provided in [`LICENSES/GPL-3.0.txt`](LICENSES/GPL-3.0.txt).

**VISTA:** The original VISTA implementation is licensed under Apache-2.0. A copy is provided in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

**WATER, WebEvo, and Similo:** No software license was identified in the inspected upstream revisions. The repository references acknowledge the original methods and implementations; they do not imply additional redistribution permissions.

The license texts above apply to the relevant third-party material. They do not, by themselves, establish a license for the entire REPATH repository.

## Dataset

The evaluation uses the Web-UI-Matching dataset released with UITestFix:

Yuanzhang Lin, Guoyao Wen, and Xiang Gao, ["Automated Fixing of Web UI Tests via Iterative Element Matching"](https://doi.org/10.1109/ASE56229.2023.00048), ASE 2023.

Dataset: [Web-UI-Matching Dataset](https://anonymous.4open.science/r/Web-UI-Dataset-9645)

The dataset is not redistributed in this repository.
