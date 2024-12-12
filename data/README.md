This folder contains the necessary code to generate the benchmarking corpus. To
generate it, install the dependency contained in the `pyproject.toml` file:

```
uv sync
```

Then, run `./generate.sh`. It will download the data and generate the corpus.
Note that the corpus is shuffled for all sizes, which implies
that the result might differ slightly from one generated corpus to another.
