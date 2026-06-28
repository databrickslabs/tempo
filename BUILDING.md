In order to build the Python wheel for installation, follow these steps:

1. Install [uv](https://docs.astral.sh/uv/) (`brew install uv`).
2. In a shell, run `cd python`.
3. Run `make build-dist`. This builds the sdist and wheel with the build backend
   pinned by cryptographic hashes (`.build-constraints.txt`).
4. Upload the resulting wheel from `python/dist/` into your environment.
