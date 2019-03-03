# Contributing


**We appreciate all kinds of help, so thank you!**

## Contributor License Agreement

We'd love to accept your code! Before we can, we have to get a few legal
requirements sorted out. By having you sign a Contributor License Agreement (CLA), we
ensure that the community is free to use your contributions.

When you contribute to the Qiskit project with a new pull request, a bot will
evaluate whether you have signed the CLA. If required, the bot will comment on
the pull request,  including a link to accept the agreement. The
[individual CLA](https://qiskit.org/license/qiskit-cla.pdf) document is
available for review as a PDF.

If you work for a company that wants to allow you to contribute your work,
then you'll need to sign a [corporate CLA](https://qiskit.org/license/qiskit-corporate-cla.pdf)
and email it to us at qiskit@qiskit.org.


## Contributing to the Project


You can contribute in many ways to this project.


### Issue Reporting

This is a good point to start, when you find a problem please add
it to the [issue tracker](https://github.com/Qiskit/qiskit-ignis/issues).
The ideal report should include the steps to reproduce it.


### Doubts Solving

To help less advanced users is another wonderful way to start. You can
help us close some opened issues. A ticket of this kind should be
labeled as ``question``.


### Improvement Proposal

If you have an idea for a new feature, please open a ticket labeled as
``enhancement``. If you could also add a piece of code with the idea
or a partial implementation, that would be awesome.


### Good First Contributions

You are welcome to contribute wherever in the code you want to, of course, but
we recommend taking a look at the "Good First Contribution" label into the
issues and pick one. We would love to mentor you!


### Doc

Review the parts of the documentation regarding the new changes and update it
if it's needed.


### Pull Requests

We use [GitHub pull requests](https://help.github.com/articles/about-pull-requests)
to accept the contributions.

A friendly reminder! We'd love to have a previous discussion about the best way to
implement the feature/bug you are contributing with. This is a good way to
improve code quality in our beloved Qiskit Ignis! So remember to file a new issue before
starting to code for a solution.

After having discussed the best way to land your changes into the codebase,
you are ready to start coding. We have two options here:

1. You think your implementation doesn't introduce a lot of code, right?. Ok,
   no problem, you are all set to create the PR once you have finished coding.
   We are waiting for it!
2. Your implementation does introduce many things in the codebase. That sounds
   great! Thanks! In this case, you can start coding and create a PR with the
   word: **[WIP]** as a prefix of the description. This means "Work In
   Progress", and allows reviewers to make micro reviews from time to time
   without waiting for the big and final solution. Otherwise, it would make
   reviewing and coming changes pretty difficult to accomplish. The reviewer
   will remove the **[WIP]** prefix from the description once the PR is ready
   to merge.


#### Pull Request Checklist


When submitting a pull request and you feel it is ready for review, please
double check that:

* The code follows the code style of the project. For convenience, you can
  execute ``make style`` and ``make lint`` locally, which will print potential
  style warnings and fixes.
* The documentation has been updated accordingly. In particular, if a function
  or class has been modified during the PR, please update the docstring
  accordingly.
* Your contribution passes the existing tests, and if developing a new feature,
  that you have added new tests that cover those changes.
* You add a new line to the ``CHANGELOG.md`` file, in the ``UNRELEASED``
  section, with the title of your pull request and its identifier (for example,
  "``Replace OldComponent with FluxCapacitor (#123)``".


#### Commit Messages


Please follow the next rules for any commit message:

- It should include a reference to the issue ID in the first line of the commit,
  **and** a brief description of the issue, so everybody knows what this ID
  actually refers to without wasting to much time on following the link to the
  issue.

- It should provide enough information for a reviewer to understand the changes
  and their relation to the rest of the code.

A good example:

```
Issue #190: Short summary of the issue
* One of the important changes
* Another important change
```

## Code

This section include some tips that will help you to push source code.

We recommend using [Python virtual environments](https://docs.python.org/3/tutorial/venv.html)
to cleanly separate Qiskit from other applications and improve your experience.


## Setup with an Environment

The simplest way to use environments is by using Anaconda

```
conda create -y -n QiskitDevenv python=3
source activate QiskitDevenv
```

In order to execute the Ignis code, after cloning the Ignis GitHub repository on your machine,
you need to have some libraries, which can be installed in this way:


```
cd qiskit-ignis
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

To better contribute to Qiskit Ignis, we recommend that you clone the Qiskit Ignis repository
and then install Qiskit Ignis from source.  This will give you the ability to inspect and extend
the latest version of the Ignis code more efficiently.  The version of Qiskit Ignis in the repository's ``master``
branch is typically ahead of the version in the Python Package Index (PyPI) repository, and
we strive to always keep Ignis in sync with the development versions of the Qiskit elements,
each available in the ``master`` branch of the corresponding repository.  Therefore,
all the Qiskit elements and relevant components should be installed from source.  This can be
correctly achieved by first uninstalling them from the Python environment in which you
have Qiskit (if they were previously installed),
using the ``pip uninstall`` command for each of them.  Next, after cloning the
[Qiskit Terra](https://github.com/Qiskit/qiskit-terra), [Qiskit Aer](https://github.com/Qiskit/qiskit-aer),
[Qiskit IBMQ Provider](https://github.com/Qiskit/qiskit-ibmq-provider) and
[Qiskit Ignis](https://github.com/Qiskit/qiskit-ignis) repositories, you can install them
from source in the same Python environment by issuing the following command repeatedly, from each of the root
directories of those repository clones: 

```
$ pip install -e .
```

exactly in the order specified above: Qiskit Terra, Qiskit Aer, Qiskit IBMQ Provider, and Qiskit Ignis.
All the other dependencies will be installed automatically.  This process may have to be repeated often
as the ``master`` branch of Ignis is updated frequently.

### Style guide

Please submit clean code and please make effort to follow existing conventions
in order to keep it as readable as possible. We use the
[Pylint](https://www.pylint.org) and [PEP
8](https://www.python.org/dev/peps/pep-0008) style guide. To ensure
your changes respect the style guidelines, run the next commands (all platforms):

```
$> cd out
out$> make lint
out$> make style
```

## Documentation


The documentation source code for the project is located in the ``docs`` directory of the general
[Qiskit repository](https://github.com/Qiskit/qiskit) and automatically rendered on the
[Qiskit documentation Web site](https://qiskit.org/documentation/). The
documentation for the Python SDK is auto-generated from Python
docstrings using [Sphinx](http://www.sphinx-doc.org. Please follow [Google's Python Style
Guide](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments)
for docstrings. A good example of the style can also be found with
[Sphinx's napolean converter
documentation](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

To generate the documentation, you need to invoke CMake first in order to generate
all specific files for our current platform.
See the [instructions](https://github.com/Qiskit/qiskit-terra/blob/master/.github/CONTRIBUTING.rst#dependencies)
in the Terra repository for details on how to install and run CMake.

## Development Cycle


Our development cycle is straightforward. Use the **Projects** board in Github
for project management and use **Milestones** in the **Issues** board for releases. The features
that we want to include in these releases will be tagged and discussed
in the project boards. Whenever a new release is close to be launched,
we'll announce it and detail what has changed since the latest version in
our Release Notes and Changelog. The channels we'll use to announce new
releases are still being discussed, but for now, you can
[follow us](https://twitter.com/qiskit) on Twitter!


### Branch Model


There are two main branches in the repository:

- ``master``

  - This is the development branch.
  - Next release is going to be developed here. For example, if the current
    latest release version is r1.0.3, the master branch version will point to
    r1.1.0 (or r2.0.0).
  - You should expect this branch to be updated very frequently.
  - Even though we are always doing our best to not push code that breaks
    things, is more likely to eventually push code that breaks something...
    we will fix it ASAP, promise :).
  - This should not be considered as a stable branch to use in production
    environments.
  - The API of Qiskit could change without prior notice.

- ``stable``

  - This is our stable release branch.
  - It's always synchronized with the latest distributed package: as for now,
    the package you can download from pip.
  - The code in this branch is well tested and should be free of errors
    (unfortunately sometimes it's not).
  - This is a stable branch (as the name suggest), meaning that you can expect
    stable software ready for production environments.
  - All the tags from the release versions are created from this branch.

### Release Cycle

From time to time, we will release brand new versions of Qiskit Terra. These
are well-tested versions of the software.

When the time for a new release has come, we will:

1. Merge the ``master`` branch with the ``stable`` branch.
2. Create a new tag with the version number in the ``stable`` branch.
3. Crate and distribute the pip package.
4. Change the ``master`` version to the next release version.
5. Announce the new version to the world!

The ``stable`` branch should only receive changes in the form of bug fixes, so the
third version number (the maintenance number: [major].[minor].[maintenance])
will increase on every new change.
