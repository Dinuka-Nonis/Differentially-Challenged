# Differentiable Programming Examples

This repository contains examples, utilities, and exercises for **differentiable programming** using **PyTorch**. The focus is on designing smooth, differentiable approximations of functions such as **step**, **absolute value**, **max**, and others that are normally non-differentiable.

These examples help you understand:

* Why smooth approximations are useful
* How gradients behave
* How PyTorch's autograd system works under the hood

---

## Folder Structure

```
DifferentiableProgramming/
│
├── differentiable/        # Differentiable function implementations
│   ├── __init__.py
│   └── soft_step.py       # Smooth approximation of the step function
│
├── tests/                 # Automated tests
│   └── test_soft_step.py  # Unit tests for soft_step
│
├── examples/              # Visualization and experimentation scripts
│   └── visualize_soft_step.py
│
└── README.md              # Project documentation
```

---

## Installation

### Requirements

* Python **3.8+**

### Install Dependencies

```bash
python -m pip install torch matplotlib pytest
```

---

## Running the Tests

Use `pytest` to run automated tests:

```bash
python -m pytest
```

The file `test_soft_step.py` checks that:

* The **soft step** function outputs correct values
* Gradients exist and can be computed through PyTorch's autograd system

Testing ensures correctness and differentiability.

---

## Running Examples

Visualize the soft step function and its gradient:

```bash
python -m examples.visualize_soft_step
```

This produces two plots:

* **Left plot:** the soft step curve (smooth 0 → 1 approximation)
* **Right plot:** the gradient `dy/dx`, showing where learning happens

Great for building intuition about differentiable approximations.

---

## Quick Explanation

### **What is Differentiable Programming?**

Designing programs and functions that are **smooth**, **continuous**, and allow for **gradient computation**, enabling optimization through backpropagation.

### **What is soft_step?**

A smooth approximation of the hard step function:

* Hard step: outputs exactly 0 or 1 (not differentiable)
* Soft step: uses a **sigmoid**, making it smooth and differentiable

### **Why Tests?**

To ensure your differentiable functions:

* Produce expected outputs
* Have valid gradients
* Behave well under autograd

### **Why Examples?**

Visualizations help you understand:

* Function shape
* Gradient shape
* Where and how a model can learn
