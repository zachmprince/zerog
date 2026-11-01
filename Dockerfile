# One stack for laptops, CI and Claude Code sessions, built from the pixi lockfile.
# Nuclear data is never baked in: mount a library and set OPENMC_CROSS_SECTIONS at run time.
FROM ghcr.io/prefix-dev/pixi:0.83.0

WORKDIR /workspace
COPY pyproject.toml pixi.lock README.md ./
COPY src ./src
RUN pixi install --locked

COPY tests ./tests
CMD ["pixi", "run", "test"]
