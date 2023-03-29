Requirements
------------

#. In order to run `varprox` package, the following libraries are required:
    * Numpy
    * Matplotlib
    * PyAFBF
    * Prox-TV
    * imagio
   They can be installed for instance with the following command

   ```
   $ pip3 install numpy matplotlib PyAFBF install prox-tv imageio
   ```

#. The documentation generator relies on Sphinx. It can be installed as follows:

   ```
   $ pip3 install sphinx pydata-sphinx-theme
   ```

Setup
-----
To run this project, install the `varprox` package locally.
For instance, you can use the following command:

```
$ python3 setup.py install --user
```

Examples
--------
The folder `examples/` contains examples of application of `varprox` module.
For example, you can run the example that estimates semivariograms:

```
$ cd examples/ 
$ python3 ./plot_afbf.py
```
