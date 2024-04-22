============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports/Feature Requests/Feedback/Questions
===============================================
It is incredibly helpful to us when users report bugs, unexpected behaviour, or request
features. You can do any one of these, or simply ask a question about how to use 21cmSense,
by filing an issue `here <https://github.com/rasg-affiliates/21cmSense/issues/new>`_.
When doing this, please try to be as succinct, but detailed, as possible, and use
a "Minimum Working Example" whenever applicable.

Documentation improvements
==========================

21cmSense could always use more documentation, whether as part of the
official 21cmSense docs, in docstrings, or even on the web in blog posts,
articles, and such.

Development
===========
Note that it is highly recommended to work in an isolated python environment with
all development requirements installed. This will also ensure that
pre-commit hooks will run that enforce the ``black`` coding style. If you do not
install these requirements, you must manually run black before committing your changes,
otherwise your changes will likely fail continuous integration.

1. First fork `21cmSense <https://github.com/rasg-affiliates/21cmSense>`_
   (look for the "Fork" button), then clone the fork locally::

    git clone git@github.com:your_name_here/21cmSense.git

2. Install the package in dev mode::

    pip install -e .[dev]

3. Install `pre-commit <https://pre-commit.com/>`_ to do style checking automatically::

    pre-commit install

4. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

5. When you're done making changes, run all the checks with `pytest <https://docs.pytest.org/en/latest/>`_::

    pytest

6. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

5. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the
pull request. You can mark the PR as a draft until you are happy for it to be merged.
