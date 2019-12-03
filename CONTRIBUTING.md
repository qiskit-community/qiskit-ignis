# Contributing

## Contributing to Qiskit Ignis

### Issue reporting

When you encounter a problem please open an issue for it to
the [issue tracker](https://github.com/Qiskit/qiskit-ignis/issues).

### Improvement proposal

If you have an idea for a new feature please open an **Feature Requestt** issue
in the [issue tracker](https://github.com/Qiskit/qiskit-ignis/issues). Opening
an issue starts a discussion with the team about your idea, how it fits in with
the project, how it can be implemented, etc.

### Code Review

Code review is done in the open and open to anyone. While only maintainers have
access to merge commits, providing feedback on pull requests is very valuable
and helpful. It is also a good mechanism to learn about the code base. You can
view a list of all open pull requests here:
https://github.com/Qiskit/qiskit-ignis/pulls
to review any open pull requests and provide feedback on it.

### Documentation

If you make a change, make sure you update the associated
*docstrings* and parts of the documentation under `docs/apidocs` that
corresponds to it. To locally build the ignis specific documentation you
can run `tox -edocs` which will compile and build the documentation locally
and save the output to `docs/_build/html`. Additionally, the Docs CI job on
azure pipelines will run this and host a zip file of the output that you can
download and view locally.

If you have an issue with the combined documentation hosted at
https://qiskit.org/documentation/ that is maintained in the
[Qiskit/qiskit](https://github.com/Qiskit/qiskit). You can also make a
[documentation issue](https://github.com/Qiskit/qiskit/issues/new/choose) if
you see doc bugs, have a new feature that needs to be documented, or think that
material could be added to the existing docs.

#### Documentation Structure

The way documentation is structured in ignis is to push as much of the actual
documentation into the docstrings as possible. This makes it easier for
additions and corrections to be made during development because the majority
of the documentation lives near the code being changed. There are 3 levels of
pieces to the normal documentation structure in ignis. The first is the rst
files in the `docs/apidocs`. These files are used to tell sphinx which modules
to include in the rendered documentation. The contain 2 pieces of information
an internal reference[1][2] to the module which can be used for internal links
inside the documentation and an `automodule` directive [3] used to parse the
module docstrings from a specified import path. For example, from terra the
dagcircuit.rst file contains:

```
.. _qiskit-dagcircuit:
.. automodule:: qiskit.dagcircuit
   :no-members:
   :no-inherited-members:
   :no-special-members:
```

The next level is the module level docstring. This docstring is at the module
level for the module specified in the `automodule` directive in the rst file.
If the module specified is a directory/namespace the docstring should be
specified in the `__init__.py` file for that directory. This module level
docstring starts to contain more details about the module being documented.
The normal structure to this module docstring is to outline all the classes and
functions of the public api that are contained in that module. This is typically
done using the `autosummary` directive[5] (or `autodoc` directives [3] directly
if the module is simple, such as in the case of `qiskit.execute`) The
autosummary directive is used to autodoc a list of different python elements
(classes, functions, etc) directly without having to manually call out the
autodoc directives for each one. This modulelevel docstring is a normally the
place you will want to provide a high level overview of what functionality is
provided by the module. This is normally done by grouping the different
components of the public API together into multiple subsections.

For example, continuing that dagcircuit module example from before the
contents of the module docstring for `qiskit/dagcircuit/__init__.py` would be:

```
"""
=======================================
DAG Circuits (:mod:`qiskit.dagcircuit`)
=======================================
.. currentmodule:: qiskit.dagcircuit
DAG Circuits
============
.. autosummary::
   :toctree: ../stubs/
   DAGCircuit
   DAGNode
Exceptions
==========
.. autosummary::
   :toctree: ../stubs/
   DAGCircuitError
"""
```

(note this is just an example and the actual module docstring for the dagcircuit
module might diverge from this)

The last level is the actual docstring for the elements listed in the module
docstring. You should strive to document thoroughly all the public interfaces
exposed using examples when necessary.

Note you can use any sphinx directive or rst formatting in a docstring as it
makes sense. For example, one common extension used is the `jupyter-execute`
directive which is used to execute a code block in jupyter and display both
the code and output. This is particularly useful for visualizations.

[1] http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html#reference-names
[2] https://www.sphinx-doc.org/en/latest/usage/restructuredtext/roles.html#ref-role
[3] http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
[4] https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#table-of-contents
[5] https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

#### Documentation Integration

The hosted documentation at https://qiskit.org/documentation/ covers the entire
qiskit project, Terra is just one component of that. As such the documentation
builds for the hosted version get built by the qiskit meta-package repository
https://github.com/Qiskit/qiskit. When commits are merged to that repo the
output of sphinx builds get uploaded to the qiskit.org website. Those sphinx
builds are configured to pull in the documentation from the version of the
qiskit elements installed by the meta-package at that point. For example, if
the meta-package version is currently 0.13.0 then that will copy the
documentation from ignis's 0.2.0 release. When the meta-package's requirements
are bumped then it will start pulling documentation from that new version. This
means if API documentation is incorrect to get it fixed it will need to be
included in a new release. Documentation fixes are valid backports for a stable
patch release per the stable branch policy (see that section below).

During the build process the contents of ignis's `docs/apidocs/` repository gets
recursively copied into a shared copy of `doc/apidocs/` in the meta-package
repository along with all the other elements. This means what is in the root of
docs/apidocs on ignis at a release will end up on the root of
https://qiskit.org/documentation/apidoc/


### Pull requests

We use [GitHub pull requests](
https://help.github.com/articles/about-pull-requests) to accept contributions.

While not required, opening a new issue about the bug you're fixing or the
feature you're working on before you open a pull request is an important step
in starting a discussion with the community about your work. The issue gives us
a place to talk about the idea and how we can work together to implement it in
the code. It also lets the community know what you're working on and if you
need help, you can use the issue to go through it with other community and team
members.

If you've written some code but need help finishing it, want to get initial
feedback on it prior to finishing it, or want to share it and discuss prior
to finishing the implementation you can open a *Work in Progress* pull request.
When you create the pull request prefix the title with the **\[WIP\]** tag (for
**W**ork **I**n **P**rogress). This will indicate to reviewers that the code in
the PR isn't in it's final state and will change. It also means that we will
not merge the commit until it is finished. You or a reviewer can remove the
[WIP] tag when the code is ready to be fully reviewed for merging.

### Contributor License Agreement

Before you can submit any code we need all contributors to sign a
contributor license agreement. By signing a contributor license
agreement (CLA) you're basically just attesting to the fact
that you are the author of the contribution and that you're freely
contributing it under the terms of the Apache-2.0 license.

When you contribute to the Qiskit Terra project with a new pull request,
a bot will evaluate whether you have signed the CLA. If required, the
bot will comment on the pull request, including a link to accept the
agreement. The [individual CLA](https://qiskit.org/license/qiskit-cla.pdf)
document is available for review as a PDF.

**Note**:
> If your contribution is part of your employment or your contribution
> is the property of your employer, then you will likely need to sign a
> [corporate CLA](https://qiskit.org/license/qiskit-corporate-cla.pdf) too and
> email it to us at <qiskit@us.ibm.com>.


### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the code style of the project and successfully
   passes the tests. For convenience, you can execute `tox` locally,
   which will run these checks and report any issues.
2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.
3. If it makes sense for your change that you have added new tests that
   cover the changes.
4. Ensure that if your change has an end user facing impact (new feature,
   deprecation, removal etc) that you have updated the CHANGELOG.md file.

### Commit messages

As important as the content of the change, is the content of the commit message
describing it. The commit message provides the context for not only code review
but also the change history in the git log. Having a detailed commit message
will make it easier for your code to be reviewed and also provide context to the
change when it's being looked at years in the future. When writing a commit
message there are some important things to remember:

* Do not assume the reviewer understands what the original problem was.

When reading an issue, after a number of back & forth comments, it is often
clear what the root cause problem is. The commit message should have a clear
statement as to what the original problem is. The bug is merely interesting
historical background on *how* the problem was identified. It should be
possible to review a proposed patch for correctness from the commit message,
 without needing to read the bug ticket.
bug ticket.

* Do not assume the code is self-evident/self-documenting.

What is self-evident to one person, might not be clear to another person. Always
document what the original problem was and how it is being fixed, for any change
except the most obvious typos, or whitespace only commits.

* Describe why a change is being made.

A common mistake is to just document how the code has been written, without
describing *why* the developer chose to do it that way. By all means describe
the overall code structure, particularly for large changes, but more importantly
describe the intent/motivation behind the changes.

* Read the commit message to see if it hints at improved code structure.

Often when describing a large commit message, it becomes obvious that a commit
should have in fact been split into 2 or more parts. Don't be afraid to go back
and rebase the change to split it up into separate pull requests.

* Ensure sufficient information to decide whether to review.

When Github sends out email alerts for new pull request submissions, there is
minimal information included, usually just the commit message and the list of
files changes. Because of the high volume of patches, commit message must
contain sufficient information for potential reviewers to find the patch that
they need to look at.

* The first commit line is the most important.

In Git commits, the first line of the commit message has special significance.
It is used as the default pull request title, email notification subject line,
git annotate messages, gitk viewer annotations, merge commit messages, and many
more places where space is at a premium. As well as summarizing the change
itself, it should take care to detail what part of the code is affected.

* Describe any limitations of the current code.

If the code being changed still has future scope for improvements, or any known
limitations, then mention these in the commit message. This demonstrates to the
reviewer that the broader picture has been considered and what tradeoffs have
been done in terms of short term goals vs. long term wishes.

* Include references to issues

If the commit fixes or is related to an issue make sure you annotate that in
the commit message. Using the syntax:

Fixes #1234

if it fixes the issue (github will close the issue when the PR merges).

The main rule to follow is:

The commit message must contain all the information required to fully
understand & review the patch for correctness. Less is not more.


### Installing Qiskit Ignis from source
Please see the [Installing Qiskit Ignis from
Source](https://qiskit.org/documentation/contributing_to_qiskit.html#installing-ignis-from-source)
section of the Qiskit documentation.


### Test

Once you've made a code change, it is important to verify that your change
does not break any existing tests and that any new tests that you've added
also run successfully. Before you open a new pull request for your change,
you'll want to run the test suite locally.

The easiest way to run the test suite is to use
[**tox**](https://tox.readthedocs.io/en/latest/#). You can install tox
with pip: `pip install -U tox`. Tox provides several advantages, but the
biggest one is that it builds an isolated virtualenv for running tests. This
means it does not pollute your system python when running. Additionally, the
environment that tox sets up matches the CI environment more closely and it
runs the tests in parallel (resulting in much faster execution). To run tests
on all installed supported python versions and lint/style checks you can simply
run `tox`. Or if you just want to run the tests once run for a specific python
version: `tox -epy37` (or replace py37 with the python version you want to use,
py35 or py36).

If you just want to run a subset of tests you can pass a selection regex to
the test runner. For example, if you want to run all tests that have "dag" in
the test id you can run: `tox -epy37 -- dag`. You can pass arguments directly to
the test runner after the bare `--`. To see all the options on test selection
you can refer to the stestr manual:
https://stestr.readthedocs.io/en/stable/MANUAL.html#test-selection

If you want to run a single test module, test class, or individual test method
you can do this faster with the `-n`/`--no-discover` option. For example:

to run a module:
```
tox -epy37 -- -n test.test_examples
```
or to run the same module by path:

```
tox -epy37 -- -n test/test_examples.py
```
to run a class:

```
tox -epy37 -- -n test.test_examples.TestPythonExamples
```
to run a method:
```
tox -epy37 -- -n test.test_examples.TestPythonExamples.test_all_examples
```


### Style guide

To enforce a consistent code style in the project we use
[Pylint](https://www.pylint.org) and
[pycodesytle](https://pycodestyle.readthedocs.io/en/latest/)
to verify that code contributions conform respect the projects
style guide. To verify that your changes conform to the style
guide you can run: `tox -elint`

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

## Development Cycle

The development cycle for qiskit-ignis is all handled in the open using
the project boards in Github for project management. We use milestones
in Github to track work for specific releases. The features or other changes
that we want to include in a release will be tagged and discussed in Github.
As we're preparing a new release we'll document what has changed since the
previous version in the release notes and Changelog.

### Branches

* `master`:

The master branch is used for development of the next version of qiskit-ignis.
It will be updated frequently and should not be considered stable. The API
can and will change on master as we introduce and refine new features.

* `stable`:
The stable branches is used to maintain the most recent released versions of
qiskit-ignis. It contains the version of the code corresponding to the latest
release for The API on these branches are stable and the only changes
merged to it are bugfixes.


### Release Cycle

From time to time, we will release brand new versions of Qiskit Terra. These
are well-tested versions of the software.

When the time for a new release has come, we will:

1. Merge the `master` branch with the `stable` branch.
2. Create a new tag with the version number in the `stable` branch.
4. Change the `master` version to the next release version.

The `stable` branch should only receive changes in the form of bug fixes.

## Stable Branch Policy

The stable branch is intended to be a safe source of fixes for high impact bugs
and security issues which have been fixed on master since a release. When
reviewing a stable branch PR we need to balance the risk of any given patch
with the value that it will provide to users of the stable branch. Only a
limited class of changes are appropriate for inclusion on the stable branch. A
large, risky patch for a major issue might make sense. As might a trivial fix
for a fairly obscure error handling case. A number of factors must be weighed
when considering a change:

- The risk of regression: even the tiniest changes carry some risk of breaking
  something and we really want to avoid regressions on the stable branch
- The user visible benefit: are we fixing something that users might actually
  notice and, if so, how important is it?
- How self-contained the fix is: if it fixes a significant issue but also
  refactors a lot of code, it's probably worth thinking about what a less
  risky fix might look like
- Whether the fix is already on master: a change must be a backport of a change
  already merged onto master, unless the change simply does not make sense on
  master.

### Backporting procedure:

When backporting a patch from master to stable we want to keep a reference to
the change on master. When you create the branch for the stable PR you can use:

`$ git cherry-pick -x $master_commit_id`

However, this only works for small self contained patches from master. If you
need to backport a subset of a larger commit (from a squashed PR for
example) from master this just need be done manually. This should be handled
by adding::

    Backported from: #master pr number

in these cases, so we can track the source of the change subset even if a
strict cherry pick doesn't make sense.

If the patch you're proposing will not cherry-pick cleanly, you can help by
resolving the conflicts yourself and proposing the resulting patch. Please keep
Conflicts lines in the commit message to help review of the stable patch.

### Backport Tags

Bugs or PRs tagged with `stable backport potential` are bugs which apply to the
stable release too and may be suitable for backporting once a fix lands in
master. Once the backport has been proposed, the tag should be removed.

The PR against the stable branch should include `[stable]` in the title, as a
sign that setting the target branch as stable was not a mistake. Also,
reference to the PR number in master that you are porting.
