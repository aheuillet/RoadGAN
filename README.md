# RoadGAN

> A project generating weather-controlled photorealistic street videos from segmentation maps.
This project borrows heavily from [Few Shot Vid2Vid](https://github.com/NVlabs/few-shot-vid2vid) and [Attribute Hallucination](https://github.com/hucvl/attribute_hallucination).

![](inference/teaser.gif)

## Requirements

- Python >= 3.6
- Pytorch >= 1.4.0
- Cuda 10.2

The full list of requirements can be found in `roadgan.yml` (conda env file).

## Installation

It is possible to install all the dependencies and download all the pre-trained models by running `install.sh` (tested on Ubuntu 18.04 and Ubuntu 20.04). This script also installs `conda` for you if it is not present on your system.

You are also entirely free to source the env file and download the pre-trained models yourself.

## Usage

To start the GUI app, run `python gui.py`.
Select the input segmentation video as well as the output save location and click on the start button to launch the conversion with the default parameters. These parameters can be changed by opening the "settings" menu and comprise:
- The weather effect (rainy, cloudy, foggy, snowy or clear)
- The time of day (day, dawn/dusk or night)
- The urban style (Boston, Berlin, Paris, Los Angeles or Beijing)