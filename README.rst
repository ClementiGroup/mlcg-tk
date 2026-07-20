mlcg-tk
=======

.. start-readme

|Docs badge| |License| |Circleci|

.. |Docs badge| image:: https://img.shields.io/badge/mlcg--tk-docs-blue.svg
   :target: https://clementigroup.github.io/mlcg-tk/

.. |License| image:: https://img.shields.io/github/license/Naereen/StrapDown.js.svg
   :target: https://opensource.org/licenses/MIT

.. |Circleci| image:: https://dl.circleci.com/status-badge/img/gh/ClementiGroup/mlcg-tk/tree/main.svg?style=shield
    :target: https://dl.circleci.com/status-badge/redirect/gh/ClementiGroup/mlcg-tk/tree/main


A collection of tools for processing raw simulation data for use in training a 
transferable coarse grained (CG) forcefield, using the MLCG library.

Installation
------------

**Before installing mlcg-tk, you must install the** `mlcg package <https://github.com/ClementiGroup/mlcg>`__.

Similar to mlcg, we encourage the use of the `uv <https://docs.astral.sh/uv/>`__ enviroment manager


Once mlcg is installed, you can install mlcg-tk as follows

.. code:: bash

  git clone git@github.com:ClementiGroup/mlcg-tk.git
  cd mlcg-tk
  uv pip install .

**For developers**

Add ``--group dev`` to install additional development dependencies
(``black``, ``pytest``, ``coverage``), e.g.

.. code:: bash

    uv pip install . --group dev

.. end-readme

How to use
----------

Extensive instructions on how to use the different tools are provided in the 
examples folder.