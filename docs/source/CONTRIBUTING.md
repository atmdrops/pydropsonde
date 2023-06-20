# Contributing to HALO-DROPS

Firstly, thanks for thinking of contributing to HALO-DROPS, and I'm hoping that you didn't land here by mistake. But even if you did, now that you are here why not help us out a bit with this package? There are [different ways](#how-can-you-help-us) in which you can contribute to HALO-DROPS. To know how to get started, jump to [Development Workflow](#development-workflow).

## How can you help us?

You can help us by:

- Submitting issues
  - Tell us about any bugs you encounter, enhancements or features that you think the package should include, or even things like some missing documentation that might be useful to you and others. Simply raise an [issue](https://github.com/Geet-George/halodrops/issues)! Even if you are not sure if something qualifies as an issue, just raise an issue anyway. Worst case, someone will point out the solution or you might figure it out yourself and help out someone in the future facing the same problem. It's win-win.
- Documenting
  - A large and very important part of HALO-DROPS is the documentation that goes with it. This includes the protocols for operations and data processing as well as the explanations behind them. We also need how-to guides to explain better how the package can be employed. And there are functions and classes that can benefit from better docstrings. In fact, this very document `CONTRIBUTING.md` needs more documentation. There is almost never enough documentation. So, pick out a small aspect of the HALO-DROPS docs and try your hand at it. The universe will thank you. Just try it.
- Coding
  - Scripts, classes, methods, tests... A lot remains to be done in the package.

Have a look at the [issues](https://github.com/Geet-George/halodrops/issues) and you will find where HALO-DROPS needs help. Pick an issue and assign it to yourself, so others know that you are working on that. If you are not sure how to proceed, you can express your interest by commenting on the issue, and someone should help you out.

## Development Workflow

1. **Fork & clone HALO-DROPS**

    Fork the [HALO-DROPS](https://github.com/Geet-George/halodrops) repository.

    Clone your fork and set the original repository to remote upstream with the following commands:

    ```bash
    git clone git@github.com:<your-github-username>/halodrops.git
    cd halodrops
    git remote add upstream git@github.com:Geet-George/halodrops.git
    ```
2. **Create the development environment**

    HALO-DROPS is developed in Python. The packages needed for development is specified in the `environment.yml` file. The package manager [conda](https://conda.io/) can be used to create an environment from the file. We recommend using [mamba](https://mamba.readthedocs.io/en/latest/installation.html), which makes the whole process quicker by much faster dependency solving and parallelizing the package downloads.

    You could also skip using conda or mamba, in favour of [pip](https://pypi.org/project/pip/), but then you would have to convert the `environment.yml` entries into a `.txt` (usually named `requirements.txt`) format that `pip` understands. There are [ways](https://gist.github.com/pemagrg1/f959c19ec18fee3ce2ff9b3b86b67c16) to do the conversion without copywriting all night, but nothing straightforward because conda is a general package manager. All that said, I recommend sticking with conda / mamba, in case HALO-DROPS decides to use non-Python packages in the future (e.g. the CLI of ASPEN)

    Be sure you are in the `halodrops` home directory, i.e. the location of the `pyproject.toml` file. This will ensure the editable installation of the `halodrops` package within the environment itself via pip (I know I said we prefer conda over pip, but [`conda develop` is a whole different story](https://github.com/conda/conda-build/issues/4251).)

    So, if you are in the right directory, let's create the environment.

    With mamba:
    ```bash
    mamba env create -f environment.yml
    ```

    > Alternatively, conda can be used (but can be slower):
    > ```bash
    > conda env create -f environment.yml
    > ```

    Activate the environment with:
    ```bash
    conda activate halodrops
    ```

    **_Get pre-commit working for you_**

    If you created the environment with the `environmnet.yml` file, then `pre-commit` should already be present in your environment. `pre-commit` is used to employ hooks for checking (and in some cases, fixing) the code before commits are made. To get pre-commit to check automatically every time you commit, use the following command:

    ```bash
    pre-commit install
    ```

    That's it. :) `pre-commit` will now parse through all hooks in the `.pre-commit-config.yaml` file and do its thing accordingly.

    Every time there is a change in the config YAML file, be sure to apply those changes for all existing files too with the following command:

    ```bash
    pre-commit run --all-files
    ```

3. **Create a branch**

    Now we have a local copy of our fork and we have the environment ready to start developing.

    It is always good coding practice to work on a different branch every time you start working on a new feature / bug-fix (yes, despite having your own fork).

    Create a branch and checkout to start working on it.
    ```bash
    git branch my-new-feature
    git checkout my-new-feature
    ```
    > Alternatively, do it all in one line.
    > ```bash
    > git checkout -b my-new-feature
    > ```
    > Note that regardless of which branch you are on currently, you can still specify off which branch you want to create a new branch by (shown here as branch off `main`):
    > ```bash
    > git branch my-new-feature main
    > ```

4. **Make your changes**

    Do your edits and push to your fork. Behold git's holy trinity!

    ```bash
    git add . # will add all uncommitted changes
    git commit -m "your commit message here" # consider giving a detailed message & not simply a header
    git push # for the first push of a branch, track it e.g. git push -u origin my-new-feature
    ```

    Every commit makes changes that are justified by one reason. The size of a commit could be a single character change or a change in thousands of lines across millions of files. But the reason behind the commit should ideally be as solitary as possible. Commit often, but not too often. Henry VIII said that.

    For making changes to the documentation, refer the {ref}`Documentation development<documentation-development>`section for steps.

5. **Submit pull request**

    Head over to Github and from the relevant branch in your fork, create a [Pull Request (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

    You can [request a PR review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) from someone. They will help with some feedback or might wholeheartedly agree with your changes. Others might also comment with their opinion. Add any necessary changes with subsequent commits to the same branch. Once everyone involved in this conversation is satisfied, the PR is merged. From personal experience though, frantically refreshing the browser every three seconds doesn't speed up the response rate of others. Sometimes, PRs can take their own sweet time.

6. **... And that's it! Thanks for helping**

(documentation-development)=
## Documentation development

HALO-DROPS uses [Sphinx](https://www.sphinx-doc.org/en/master/index.html) with the [Book theme](https://sphinx-book-theme.readthedocs.io/en/stable/) to display its documentation, and is hosted by [ReadTheDocs](https://readthedocs.org/). All documentation comes from markdown files or Jupyter notebooks, except the API reference, which is built automatically from the docstrings in the modules in the `src` directory, thanks to [sphinx-autodoc2](https://sphinx-autodoc2.readthedocs.io/en/latest/).

### Steps to make documentation changes

1. You'll find the source files for documentation in the `docs/source/` directory. If you want to change documentation in the API reference, then head over to the corresponding module in the source code and change the relevant docstring.

2. Make the change. Here are some referencing tips for both Markdown files and for docstrings.

   - For cross-referencing within the document, use the `myst` style, which is:
      ```
      {ref}`Text you want displayed <reference-tag>`
      ```
      The `reference-tag` is added to the relevant sub-header as follows:
      ```
      (reference-tag)=
      ## The Header
      ```

   - For cross-referencing a different document, use:
      ```
      {doc}`Text you want displayed <path/to/doc/FilenameWithoutExtension>`
      ```

   - For URLs, simple markdown cross-referencing should suffice, e.g.
      ```
      [Text you want displayed](https://thereference.url)
      ```



3. Rebuild the documentation with:

```bash
sphinx-build -n docs/source docs/build
```

The `-n` flag is to enable [nitpicky mode](https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpicky), so that we catch all warnings with missing references.

For automatically updating the documentation upon any changes you make, we also have the [`sphinx-autobuild`](https://pypi.org/project/sphinx-autobuild/) installed in our development environment. This detects any changes to files in the `docs/source`  directory, rebuilds the docs and starts a server at http://127.0.0.1:8000 to display the rebuilt docs. Stop the server with `Ctrl + C`.

```bash
sphinx-autobuild  -n docs/source/ docs/build/
```
There are several arguments of `sphinx-build` & `sphinx-autobuild` that can be used to make modifications to the workflow. For example, the `--open-browser` argument will also open the browser to the port.
