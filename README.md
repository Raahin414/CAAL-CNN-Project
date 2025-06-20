# Vectorized Convolutional Neural Network in RISC-V

## Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) on the RISC-V 32-bit architecture, making use of the RISC-V Vector Extension (RVV). Built entirely in assembly language, the CNN processes grayscale digit images from the MNIST dataset and classifies them into one of ten digit classes (0–9).

The CNN is structured to run on a custom RISC-V simulation environment (VEER ISS), and includes both scalar and vectorized implementations of major neural network components.

## Table of Contents

- Overview
- Installation
- Usage
- Project Structure
- Architecture Details
- Challenges
- Acknowledgments

## Installation

To run this project, you will need the following tools:

- RISC-V Toolchain – https://github.com/riscv-collab/riscv-gnu-toolchain  
- VEER ISS (Imperas) – RISC-V simulator  
- Venus Simulator (for debugging scalar RISC-V instructions online)

We recommend running these tools on a Linux VM (e.g., Ubuntu) for compatibility.

## Usage

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Raahin414/CAAL-CNN-Project.git
cd CAAL-CNN-Project
