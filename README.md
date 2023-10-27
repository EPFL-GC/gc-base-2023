CS 457 Geometric Computing — Assignments
======================================

## Build status
insert your build badge URL here

## Installation notes 

You can set the environment up by running the following command:

```
conda env create -f environment.yml
conda activate gc_course_env
```

Alternatively, the following commands should produce the same outcome:

```
conda create --name gc_course_env python=3.9
conda activate gc_course_env
conda install -c conda-forge gmsh jupyterlab pytest pytest-html pytest-timeout matplotlib svgwrite meshplot triangle
pip install pyvista libigl opencv-python
```

For HW2 on using MacOS, please install [PyTorch](https://pytorch.org/get-started/locally/) as follow

```
conda install pytorch::pytorch=2.0 -c pytorch
```

For Linux and Windows user, please run

```
conda install pytorch=2.0 cpuonly -c pytorch
```
