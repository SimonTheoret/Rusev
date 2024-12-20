This folder contains the necessary code to generate the benchmarking corpus. It
the following dependencies:

``` toml
    datasets>=3.1.0,
    fire>=0.7.0,
    jsonlines>=4.0.0,
    torch>=2.5.1,
    transformers>=4.46.3,
```

 To easily generate the virtual env, install the dependency contained in the
 `pyproject.toml` file:

``` sh
uv sync
```
This commands uses the file `uv.lock` to reproduce exactly the environement locally.

Then, run `./generate.sh`. It will download the data and generate the corpus.
Note that the corpus is shuffled for all sizes, which implies that the result
might differ slightly from one generated corpus to another.
