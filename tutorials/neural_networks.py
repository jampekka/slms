import marimo

__generated_with = "0.6.14"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        rf"""
        # Neural network models

        **TODO**, please check back later.
        """
    )
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
