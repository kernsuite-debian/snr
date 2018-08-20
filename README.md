# SNR

Some signal to noise computation kernels for many-core accelerators, with classes to use them in C++.

## Publications

* Alessio Sclocco, Joeri van Leeuwen, Henri E. Bal, Rob V. van Nieuwpoort. _A Real-Time Radio Transient Pipeline for ARTS._ **3rd IEEE Global Conference on Signal & Information Processing**, December 14-16, 2015, Orlando (Florida), USA. ([print](http://www.sciencedirect.com/science/article/pii/S2213133716000020)) ([preprint](http://alessio.sclocco.eu/pubs/sclocco2015a.pdf)) ([slides](http://alessio.sclocco.eu/pubs/Presentation_GlobalSIP2015.pdf))
* Alessio Sclocco, Henri E. Bal, Rob V. van Nieuwpoort. _Finding Pulsars in Real-Time_. **IEEE International Conference on eScience**, 31 August - 4 September, 2015, Munic, Germany. ([print](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7304280)) ([preprint](http://alessio.sclocco.eu/pubs/sclocco2015.pdf)) ([slides](http://alessio.sclocco.eu/pubs/Presentation_eScience2015.pdf))

# Installation

Set the `INSTALL_ROOT` environment variable to the location of the pipeline sourcode.
If this package is installed in `$HOME/Code/APERTIF/Dedispersion` this would be:

```bash
 $ export INSTALL_ROOT=$HOME/Code/APERTIF
```

Then build and test as follows:

```bash
 $ make install
```

## Dependencies

* [utils](https://github.com/isazi/utils) - master branch
* [OpenCL](https://github.com/isazi/OpenCL) - master branch
* [AstroData](https://github.com/isazi/AstroData) - master branch

# Included programs

The integration step is typically compiled as part of a larger pipeline, but this repo contains two example programs in the `bin/` directory to test and autotune an integration kernel.

## SNRTest 

Checks if the output of the CPU is the same for the GPU.
The CPU is assumed to be always correct.
Takes platform, layout, and kernel arguments, and has the following extra parameters:

 * *print_code*     Print kernel source code
 * *print_results*  Prints the integrated data

TODO: *samples_dms* and *dms_samples* options?

## SNRTuning

Tune the SNR kernel's parameters by doing a complete sampling of the parameter space.
Kernel configuration and runtime statistics are written to stdout.
Takes platform, layout, and tuning arguments.

The output can be analyzed using the python scripts in in the *analysis* directory.

## printCode

Prints the code for a specific integration kernel to stdout.
Takes platform, data layout, and kernel arguments.

## Commandline arguments

Description of common commandline arguments for the separate binaries.

### Compute platform specific arguments

 * *opencl_platform*     OpenCL platform
 * *opencl_device*       OpenCL device number
 * *padding*             number of elements in the cacheline of the platform

### Data layout arguments

 * *samples*       Number of samples; ie length of time dimension
 * *dms*           Number of dispersion measures; ie length second dimension
 * *dms_samples*   Ordering of the two dimensions: samples is fastest
 * *samples_dms*   Ordering of the two dimensions: dms is fastest

### Tuning parameters

 * *iterations*    Number of times to run a specific kernel to improve statistics.
 * *min_threads*   Minimum number of threads
 * *max_threads*   Maximum number of threads
 * *max_items*     Maximum number of variables that the automated code is allowed to use.

### Kernel Configuration arguments

 * *threads0*    TODO
 * *items0*      TODO

## Dependencies

* [AstroData](https://github.com/isazi/AstroData) - master branch
* [OpenCL](https://github.com/isazi/OpenCL) - master branch
* [utils](https://github.com/isazi/utils) - master branch

# Analyzing tuning output

Kernel statistics can be saved to a database, and analyzed to find the optimal configuration.

## Setup

### MariaDB

Install mariadb, fi. via your package manager. Then:

0. log in to the database: ` $ mysql`
1. create a database to hold our tuning data: `create database AAALERT`
2. make sure we can use it (replace USER with your username): `grant all privileges on AAALERT.* to 'USER'@'localhost';` 
3. copy the template configuration file: `cp analysis/config.py.orig analysis/config.py` and enter your configuration.

### Python

The analysis scripts use some python3 packages. An easy way to set this up is using `virtualenv`:

```bash
$ cd $INSTALL_ROOT/SNR/analysis`
$ virtualenv --system-site-packages --python=python3 env`
$ . env/bin/activate`
```

And then install the missing packages:

```bash
$ pip install pymysql
```

## Run Analysis

The analysis is controlled by the `analysis/dedispersin.py` script.
It prints data as space-separated data to stdout, where you can plot it with fi. gnuplot, or copy-paste it in your favorite spreadsheet.
You can also write it to a file, that can then be read by the SNR code.

1. List current tables: `./snr.py list`
2. Create a table: `./snr.py create <table name>`
3. Enter a file create with SNRTuning into the database: `./snr.py load <table name> <file name>`
4. Find optimal kernel configuration: `./snr.py tune <table name> max <channels> <samples>`
     
The tune subcommand also takes a number of different parameters: `./snr.py tune <table> <operator> <channels> <samples>`

 * operator: max, min, avg, std  (SQL aggergation commands)
 * channels: number of channels
 * samples: number of samples

## License

Licensed under the Apache License, Version 2.0.

