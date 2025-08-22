# xRHEED


<p align="left">
	<img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12">
	<img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Status: Alpha">
	<img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
	<img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
	<img src="https://img.shields.io/badge/linter-ruff-0C7A5B.svg" alt="Ruff">
	<img src="https://img.shields.io/badge/uv-compatible-blueviolet.svg" alt="UV">
	<a href="https://xrheed.readthedocs.io/en/latest/api.html"><img src="https://img.shields.io/badge/documentation-API-blue.svg" alt="API Documentation"></a>
</p>

| Branch  | Tests Status | Docs Status |
|---------|--------------|-------------|
| main    | ![Tests & Docs](https://github.com/mkopciuszynski/xrheed/actions/workflows/ci.yml/badge.svg?branch=main) | ![Docs](https://github.com/mkopciuszynski/xrheed/actions/workflows/ci.yml/badge.svg?branch=main) |
| dev     | ![Tests & Docs](https://github.com/mkopciuszynski/xrheed/actions/workflows/ci.yml/badge.svg?branch=dev) | ![Docs](https://github.com/mkopciuszynski/xrheed/actions/workflows/ci.yml/badge.svg?branch=dev) |

An xarray-based toolkit for RHEED images analysis

## What is RHEED?

**RHEED** stands for **Reflection High Energy Electron Diffraction**. It is an experimental technique used to monitor and control the quality of crystal surfaces. In RHEED, a high-energy electron beam (typically around 20 keV) strikes the crystal surface at a very low incident angle (usually below 5 degrees). As a result, RHEED is extremely surface-sensitive and typically probes only a few atomic layers at the surface of the crystal.

## Project Goals

The goal of this software is to provide a flexible toolkit for RHEED image analysis. It is designed to help with:

- Loading and preparing RHEED images
- Creating and analyzing intensity profiles
- Overlaying theoretically predicted diffraction spot positions (using kinematic theory and Ewald construction)

## Usage

This package is not a GUI application. Instead, it provides a set of tools and functions for data loading, processing, and analysis, intended to be used in scripts or interactive environments (such as Jupyter notebooks).

With xrheed, you can efficiently prepare your RHEED data for further scientific analysis and visualization.
