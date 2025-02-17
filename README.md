# 3DSMC_BA
This is the repository for the project about Bundle Adjustment in the class 3DSMC.

## Requirements
This project requires:
- `CMake` for building
- `Ceres` for solving the Bundle Adjustment problem
- `Eigen` as a matrix backend (expected to be in 'Libs/Eigen', but can be changed in the cmake-file)
- `OpenCV` for feature extraction, matching, and various other tasks
- `TIFF` for OpenCV
- [Bundle Fusion datasets](https://graphics.stanford.edu/projects/bundlefusion/index.html#data)
- [Bundle Adjusment in the Large datasets](https://grail.cs.washington.edu/projects/bal/)

## Installation
Just download the repository and run from within the build directory
```
cmake ..
make
```

## Usage
The built binary `./bundle_adjustment` can be run to get a simple help message explaining the usage. Possible parameters are:
- `--type <type>`, the type of dataset - 'bal' for Bundle Adjustment in the Large, 'bf' for Bundle Fusion
- `--path <path>`, path to data-directory (Bundle Fusion) or to problem.txt (BAL)
- `--outliers <double>`, factor of outliers that should be removed when optimizing based on mean distance, default `2.0`
- `--iterations <int>`, maximum number of iterations the solver should optimize, default `20`
- Bundle Fusion Specific:
- `--nimg`, number of images to load, default `10`
- `--every <int>`, loads every i-th image of the dataset, default `1` aka every image
- `--matches <int>`, limit of matches that should be used for triangulation (to prevent over constraining), default `20`
- `--compare-window <int>`, compares each image to the `window_size` following images, default `1`
- `--full-comparison`, if used, every image will be compaired to every other image
- `--use-poses`, if used, instead of pose estimation the datasets given camera poses will be used

Assuming the `office3` dataset of Bundle Fusion is located at `Data/BF/office3` (i. e. all images etc. are stored there) one could run from within the `build` directory:
```
./bundle_adjustment --type bf --path "../Data/BF/office3" --nimg 50 --compare-window 2 --use-poses
```
Assuming the `problem-138-44033-pre.txt` file of the Trafalgar dataset of BAL is located at `Data/BAL/Trafalgar/problem-138-44033-pre.txt` one could run from within the `build` directory:
```
./bundle_adjustment --type bal --path "../Data/BAL/Trafalgar/problem-138-44033-pre.txt" --iterations 30 --outliers 1.5
```
