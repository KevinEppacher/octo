

export PYTHONPATH=$PYTHONPATH:~/Documents/Master/3_Semester/Modern_Robot_Concepts/octo

/home/kevin/Documents/Master/3_Semester/Modern_Robot_Concepts/octo/examples/07_example.py:1: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  import pkg_resources
gym: 0.26.2
numpy: 1.24.3
ml_dtypes: 0.5.0
chex: 0.1.85
optax: 0.1.5
tensorflow_probability: 0.23.0
tensorflow: 2.15.0
jax: 0.4.20
distrax: 0.1.5
flax: 0.7.5
ml_collections: 1.0.0
tqdm: 4.67.1
absl-py: 2.1.0
scipy: 1.12.0
wandb: 0.19.1
einops: 0.8.0
imageio: 2.36.1
moviepy: 1.0.3
pre-commit: 3.3.3
transformers: 4.47.1
tensorflow_hub: 0.16.1
tensorflow_text: 2.15.0
tensorflow_datasets: 4.9.2
tensorflow_graphics: 2021.12.3
plotly: 5.24.1
matplotlib: 3.10.0
(octo) kevin@kevin:~/Documents/Master/3_Semester/Modern_Robot_Concepts/octo$ 

# Troubleshoot Websites
## 1
https://github.com/octo-models/octo/issues/125

Resolution: make sure your jaxlib matches the version of jax (jax==0.4.20 in the requirement.txt)
Run
pip install jaxlib====0.4.20

## 2

https://github.com/octo-models/octo/issues/71

Thank you for your great work! I encountered some dependency conflicts when installing the project environment, where the version of scipy should be earlier than the latest 1.13.0 (2024.4.2). Otherwise, problems "AttributeError: module 'scipy.linalg' has no attribute 'tril'" will occur. This can be solved by returning the scipy version to 1.12.0. Hope you can fix the dependency version in requirement.txt, which is scipy<1.6.0,>=1.12.0. Thank you very much!

pip install scipy==1.12.0

# Usage

(octo) kevin@kevin:~/Documents/Master/3_Semester/Modern_Robot_Concepts/octo$ python examples/07_example.py 