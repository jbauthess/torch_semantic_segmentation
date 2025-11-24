# Torch Semantic Segmentation
The goal of this project is to implement main semantic segmentation models using pytorch and to propose tools to easily train and evaluate those models. 

## Installation

To obtain the best performances, the pytorch library installation may depend of your computer hardware configuration (GPU available?). 
For this reason, torch and torchvision are intentionally omitted of the pyproject.toml file and need to be installed separately to get the more suitable version.
To get the \<url\> of the adequat source for installing pytorch library corresponding to your configuration, please follow instructions here:
[torch](https://pytorch.org/get-started/locally/)


Then run:
```
poetry install
poetry run pip install torch torchvision --index-url <url> 
```

