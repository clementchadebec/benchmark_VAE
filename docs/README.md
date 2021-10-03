**The latest version of the documentation is available on [readthedocs](https://pythae.readthedocs.io/en/latest/)** 

To generate the documentation locally, do the following:

- Make sur you installed the requirement at the root of this repo
    ```
    pip install -r ../requirements.txt
    ```

- Install [sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) with pip and dependencies

    ```
    pip install sphinx
    pip install sphinx_rtd_theme
    pip install sphinxcontrib.bibtex
    ```

- Then build the documentation:

    ```
    sphinx-build -b html source build
    ```

**note** do not pay attention the logs warnings

Then go to `build` folder and open the `intex.html` file (in a browser for example)
