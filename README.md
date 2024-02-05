**We are actively working on the documentation**

# mmv-HITL4TRK

[![License](https://img.shields.io/pypi/l/mmv_hitl4trk.svg?color=green)](https://github.com/MMV-Lab/mmv_hitl4trk/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mmv_hitl4trk.svg?color=green)](https://pypi.org/project/mmv_hitl4trk)
[![Python Version](https://img.shields.io/pypi/pyversions/mmv_hitl4trk.svg?color=green)](https://python.org)
[![tests](https://github.com/MMV-Lab/mmv_hitl4trk/workflows/tests/badge.svg)](https://github.com/MMV-Lab/mmv_hitl4trk/actions)
[![codecov](https://codecov.io/gh/MMV-Lab/mmv_hitl4trk/branch/main/graph/badge.svg)](https://codecov.io/gh/MMV-Lab/mmv_hitl4trk)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/mmv_hitl4trk)](https://napari-hub.org/plugins/mmv_hitl4trk)

A plugin to use with napari to segment and track cells via HumanInTheLoop(HITL)-approach.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## Usage
Load a zarr-file consisting of Image, Label and Tracks layer.

## Installation

You can install `mmv_hitl4trk` via [pip]:

    pip install mmv_hitl4trk


By default, CPU is used for segmentation computing. We have tried to optimize the CPU computing time, but still recommend GPU computing. For more detailed instructions on how to install GPU support look [here](https://github.com/MouseLand/cellpose#gpu-version-cuda-on-windows-or-linux).

<!-- 

To install latest development version :

    pip install git+https://github.com/MMV-Lab/mmv_hitl4trk.git -->


## Documentation
This plugin was developed to analyze 2D cell migration. It includes the function of segmenting 2D videos using [Cellpose](https://github.com/MouseLand/cellpose) (both CPU and GPU implemented) and then tracking them using different automatic tracking algorithms, depending on the use case. For both segmentation and tracking, we have implemented user-friendly options for manual curation after automatic processing. In conjunction with napari's inherent functionalities, our plugin provides the capability to automatically track data and subsequently process the tracks in three different ways based on the reliability of the automated results. Firstly, any potentially existing incorrect tracks can be rectified in a user-friendly manner, thereby maximizing the evaluation of available information. Secondly, unreliable tracks can be selectively deleted, and thirdly, individual tracks can be manually or semi-automatically created for particularly challenging data, ensuring reliable results. In essence, our tool aims to offer a valuable supplement to the existing fully automated tracking tools and a user-friendly means to analyze videos where fully automated tracking has been previously challenging.

Common metrics such as speed, cell size, velocity, etc... can then be extracted, plotted and exported from the tracks obtained in this way. Furthermore, the plugin incorporates a functionality to assess the automatic tracking outcomes using a [quality score](https://doi.org/10.1371/journal.pone.0144959). Since automated tracking may not be consistently 100% accurate, presenting a quality measure alongside scientific discoveries becomes essential. This supplementary metric offers researchers valuable insights into the dependability of the produced tracking results, fostering informed data interpretation and decision-making in the analysis of cell migration.

More detailed information and instructions on each topic can be found in the following sections.

### Get started

To load raw data, you can simply drag & drop them into napari. Ensure that the 'Image' combobox on the right displays the correct layer afterward. To load custom segmentations, you can do equivalent.

The "save as" button can be used to save the existing layers (raw, segmentation, tracks) in a single .zarr file, which can be loaded again later using the "load" button. The "save" button overwrites the loaded .zarr file.

The computation mode is used to set how many of the available CPU cores (40% or 80%) are to be used for computing the CPU segmentation and tracking and therefore has a direct impact on performance.


### Segmentation

For segmentation, we use the state of the art instance segmentation method Cellpose. We provide one (/several??) model that we trained and has proven successful for our data ([see more information](https://doi.org/10.1038/s41467-023-43765-3)).

(...)



#### Automatic instance segmentation

(...)

##### Custom models

The plugin supports loading custom Cellpose models. To train your own Cellpose model, [this](https://cellpose.readthedocs.io/en/latest/train.html) might be helpful.

In future versions, we plan to support fine-tuning of Cellpose models within the plugin. 


#### Manual curation

Be aware that removing a cell cuts the track the cell is on. Resulting tracks shorter than two timesteps will be removed

(...)

### Tracking

The plugin supports both coordinate-based and overlap-based tracking. Overlap-based tracking requires more computation, but can also be used in particularly complicated data for individual cells.
In our experience, coordinate-based tracking has proven itself in cases with reliable segmentation. Overlap-based tracking serves as a useful complement in cases where the segmentation is not of sufficient quality.

(...)

#### Manual curation

(...)

### Analysis

The plugin supports the calculation of various metrics, which can be divided into two categories: migration-based (such as speed, direction, ...) and cell-based (such as size, eccentricity, ...). Through the use of these metrics, a comprehensive understanding of the available data can be obtained.

(...)

### Evaluation

To be aware of the accuracy of your automatic tracking and segmentation results, we have implemented an evaluation function based on [this manuscript](https://doi.org/10.1371/journal.pone.0144959). Evaluation is always carried out against the latest results of automatic segmentation and automatic tracking or previously created results loaded via the plugin's own load function. We may implement the option to evaluate external segmentations in the future, but for now you can use save and load as a workaround.

To evaluate results, at least 2 consecutive frames must first be corrected manually. The plugin saves the previously mentioned automatic or loaded results in the background, so no activation via button or similar is necessary before manual correction.

(...)


#### Segmentation evaluation

(...)

#### Tracking evaluation

(...)

false positives:
	check if cell from eval has a match > .4 IoU. If no, check if cell has second highest match >= .2 IoU. If no, then fp
	
false negatives:
	check if cell from gt has a match > .4. If no, then fn.
	check if matched cell maxIoU is higher than match. If yes, then fn.
	check if matched cell top 2 maxIoU are equal. If yes, then half fn (this will apply for both cells)
	
split cell:
	check if cell from eval has more than one match, and if second highest match is >= .2 IoU. If yes, then sc
	
added edge:
	check if a connection in gt has both cells matched in eval & the matched cells are connected. if no, then ae
	
deleted edge:
	check if a connection in eval has both cells matched in gt & the matched cells are connected. if no, then de


## Hotkeys

Here's an overview of the hotkeys. All of them can also be found in the corresponding tooltips. 

- `E` - Load next free segmentation ID
- `S` - Overlap-based single cell tracking 

## Development plan

We will continue to develop the plugin and implement new features in the future. Some of our plans in arbitrary order:

- Support of lineages
- Support training custom Cellpose models within the plugin
- Model optimization to further optimize segmentation computation
- Support evaluation of external segmentations
- ...

If you have a feature request, please [file an issue].

## Resources

The following resources may be of interest:

- [Cellpose]()

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"mmv_hitl4trk" is free and open source software

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

[file an issue]: https://github.com/MMV-Lab/mmv_hitl4trk/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
