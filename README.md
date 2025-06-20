# Vectorized Convolutional Neural Network in RISC-V

## Overview

This project demonstrates the implementation of a Convolutional Neural Network (CNN) on the RISC-V 32-bit architecture using the RISC-V Vector Extension (RVV). Built entirely in assembly language, the CNN classifies grayscale digit images from the MNIST dataset (0–9) without relying on any high-level libraries.

We implemented each CNN layer from scratch, combining scalar and vectorized RISC-V instructions to achieve a functioning digit classifier.

---

## Table of Contents

- [Overview](#overview)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Architecture Details](#architecture-details)  
- [Challenges](#challenges)  
- [Acknowledgments](#acknowledgments)  

---

## Installation

To run the project, install the following:

- [RISC-V GNU Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)  
- VEER ISS simulator (from Imperas or SiFive)  
- Venus RISC-V simulator (for quick scalar layer testing)  
- Python 3 (for preprocessing scripts)

We recommend using a Linux VM (e.g., Ubuntu) for smooth operation with VEER ISS and file system compatibility.

---

## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/Raahin414/CAAL-CNN-Project.git
cd CAAL-CNN-Project
````

2. **(Optional) Preprocess inputs and weights:**

```bash
cd python
python format_image.py          # Converts image into a usable 28x28 buffer
python extract_weights.py       # Extracts trained weights and biases
cd ..
```

3. **Build and run the CNN using VEER ISS:**

```bash
make allV
```

4. **Check the output:**

```bash
cat build/log.txt
```

You’ll see 10 float values — these are probabilities for digits 0 through 9. The highest value is the predicted digit.

> Example:
> `0.0001 0.0000 0.0000 0.0000 0.0002 0.8900 0.1083 0.0000 0.0013 0.0000`
> → The model predicts digit **5** with \~89% confidence.

---

## Project Structure

```plaintext
.
├── src/                     - RISC-V assembly source files
│   ├── cnn.s                - Full CNN pipeline
│   ├── conv_vector.s        - Vectorized convolution layer
│   ├── relu_maxpool.s       - Integrated ReLU + MaxPooling
│   ├── flatten.s            - Flatten layer
│   ├── dense.s              - Fully connected layer
│   ├── softmax.s            - Taylor-approximated softmax
│
├── python/                  - Python helpers
│   ├── format_image.py      - Converts MNIST image
│   ├── extract_weights.py   - Extracts trained weights/biases
│
├── cpp/                     - Reference C++ implementations
│   └── conv.cpp             - C++ convolution baseline
│
├── build.sh                 - Shell script to assemble & simulate
├── Makefile                 - Build configuration
└── README.md
```

---

## Architecture Details

Our CNN pipeline consists of the following layers, each written manually in RISC-V assembly:

1. **Input (28×28):**
   A grayscale image of a digit is passed as a 2D float matrix.

2. **Convolution Layer (5×5 filters × 8):**

   * Produces 8 feature maps of size 24×24
   * Uses `vle32.v`, `vfmacc.vv`, and bias loading with `vfmv.v.f`
   * Tail handling is skipped for simplicity

3. **ReLU Activation:**

   * Zeroes out negative values using `fmax.s fa0, fa0, zero`

4. **MaxPooling (2×2, stride 2):**

   * Reduces each 24×24 feature map to 12×12
   * Processes blocks using scalar and unrolled vector instructions

5. **Flatten Layer:**

   * Converts 3D tensor (8×12×12) into 1×1152 vector
   * Uses `vlse32.v` for strided channel-first loading, `vse32.v` to store in channel-last

6. **Dense Layer (1152×10):**

   * Performs matrix-vector multiplication
   * Uses `vle32.v`, `vfmacc.vv`, and `vfredsum.vs` for reduction

7. **Softmax Layer:**

   * Applies Taylor series expansion to approximate `exp(x)`
   * Computes normalized probabilities for 10 classes

---

## Challenges

* **Limited RVV support:** Instructions like `vmerge` and `vslt` were unsupported on VEER ISS, requiring alternate logic.
* **Manual math:** No built-in exponential function meant implementing `exp(x)` using Taylor expansion for softmax.
* **Debugging:** Bugs in early layers caused downstream issues, requiring repeated testing and patching across layers.
* **Performance:** Vectorization improved speed but made debugging harder, especially with memory alignment and loop edge cases.
* **VM Instability:** Running VEER ISS in a virtual machine often caused lag and crashes when testing large inputs.

---

## Acknowledgments

This project was created for the **Computer Architecture and Assembly Language** course at **IBA Karachi**.

### Team Members

* **Muhammad Hamza Arif**
* **Haris Khalid**
* **Raahin Tajuddin**
* **Zuhair Amirali Merchant**

Special thanks to our instructor for guidance and the VEER ISS resources that made this project possible.

