from pathlib import Path
from types import FunctionType, SimpleNamespace
from analysis.common import study as c
from analysis.classification import validate_prompts as original
from analysis.common import cached_retrieval as retrieval


def main():

    def save(path, value):
        value["source_hashes"][c.rel(Path(original.__file__))] = c.sha(
            Path(original.__file__)
        )
        value["source_hashes"][c.rel(Path(__file__))] = c.sha(Path(__file__))
        c.save(path, value)

    proxy = SimpleNamespace(**vars(c) | {"save": save})
    FunctionType(original.main.__code__, vars(original) | {"c": proxy, "r": retrieval})(
        "broad"
    )


if __name__ == "__main__":
    with c.offline():
        main()
