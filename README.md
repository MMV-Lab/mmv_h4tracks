We are actively working on the documentation

# mmv-tracking-napari

[![License](https://img.shields.io/pypi/l/mmv-tracking-napari.svg?color=green)](https://github.com/MMV-Lab/mmv-tracking-napari/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mmv-tracking-napari.svg?color=green)](https://pypi.org/project/mmv-tracking-napari)
[![Python Version](https://img.shields.io/pypi/pyversions/mmv-tracking-napari.svg?color=green)](https://python.org)
[![tests](https://github.com/MMV-Lab/mmv-tracking-napari/workflows/tests/badge.svg)](https://github.com/MMV-Lab/mmv-tracking-napari/actions)
[![codecov](https://codecov.io/gh/MMV-Lab/mmv-tracking-napari/branch/main/graph/badge.svg)](https://codecov.io/gh/MMV-Lab/mmv-tracking-napari)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/mmv-tracking-napari)](https://napari-hub.org/plugins/mmv-tracking-napari)

A plugin to use with napari to segment and track cells via HumanInTheLoop(HITL)-approach.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## Usage
Load a zarr-file consisting of Image, Label and Tracks layer.

## Installation

You can install `mmv-tracking-napari` via [pip]:

    pip install mmv-tracking-napari

<!-- 

To install latest development version :

    pip install git+https://github.com/MMV-Lab/mmv-tracking-napari.git -->


## Documentation
This plugin was developed to analyze 2D cell migration. It includes the function of segmenting 2D+T videos using [Cellpose](https://github.com/MouseLand/cellpose) (both CPU and GPU implemented) and then tracking them using different automatic tracking algorithms, depending on the use case. For both segmentation and tracking, we have implemented user-friendly options for manual curation after automatic processing. In conjunction with napari's inherent functionalities, our plugin provides the capability to automatically track data and subsequently process the tracks in three different ways based on the reliability of the automated results. Firstly, any potentially existing incorrect tracks can be rectified in a user-friendly manner, thereby maximizing the evaluation of available information. Secondly, unreliable tracks can be selectively deleted, and thirdly, individual tracks can be manually or semi-automatically created for particularly challenging data, ensuring reliable results. In essence, our tool aims to offer a valuable supplement to the existing fully automated tracking tools and a user-friendly means to analyze videos where fully automated tracking has been previously challenging.

Common metrics such as speed, cell size, velocity, etc... can then be extracted, plotted and exported from the tracks obtained in this way. Furthermore, the plugin incorporates a functionality to assess the automatic tracking outcomes using a [quality score](https://doi.org/10.1371/journal.pone.0144959). Since automated tracking may not be consistently 100% accurate, presenting a quality measure alongside scientific discoveries becomes essential. This supplementary metric offers researchers valuable insights into the dependability of the produced tracking results, fostering informed data interpretation and decision-making in the analysis of cell migration.

More detailed information and instructions on each topic can be found in the following sections.

### Segmentation

### Tracking

### Analysis

### Evaluation


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"mmv-tracking-napari" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/MMV-Lab/mmv-tracking-napari/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## Notes

false positives:
	check if cell from eval has a match > .4 IoU. If no, check if cell has second highest match >= .2 IoU. If no, then fp
	
false negatives:
	check if cell from gt has a match > .4. If no, then fn 
	check if matched cell maxIoU is higher than match. If yes, then fn
	ckeck if matched cell top 2 maxIoU are equal. If yes, then half fn (this will apply for both cells)
	
split cell:
	check if cell from eval has more than one match, and if second highest match is >= .2 IoU. If yes, then sc
	
added edge:
	check if a connection in gt has both cells matched in eval & the matched cells are connected. if no, then ae
	
deleted edge:
	check if a connection in eval has both cells matched in gt & the matched cells are connected. if no, then de
