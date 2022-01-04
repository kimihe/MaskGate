import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="pass-mask-gate",
  version="0.0.1",
  author="PAN, Jun",
  author_email="sparkpanjun@gmail.com",
  description="A learnable mask generate for patch skip",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/useame/MaskGate",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
  "Operating System :: OS Independent",
  ],
)