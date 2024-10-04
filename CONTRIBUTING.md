# Contributing Guide

In thise course, we follow an open source development approach, and highly encourage and appreciate any kind of contributions. 👐

For example, you can help by improving the documentation, setup guides, the lecture notes, provide bug fixes for the assignments, work on open issues (watch out for the [good first issue label](https://github.com/pkeilbach/htwg-practical-nlp/labels/good%20first%20issue)) or just fix some typos.
However small it is, every contribution is welcome - and will probably be rewarded (TBD)! 🏆

This guide will help you to get started with your first contribution! 🚀

## Forking the Course Repository

The recommended way to work with this course is to [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks) the course repository.

While cloning, as described in the [getting started guide](./docs/getting_started.md#clone-the-repository), lets you _participate_ in the course and work on the assignments, only forking the course repository allows you to contribute back bug fixes, enhancements, etc.

To fork this repository, you can follow the official GitHub documentation on how to [fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

Cloning and setting up the environment works as described in the [getting started guide](./docs/getting_started.md#clone-the-repository), besides that you need to clone your fork:

```sh
git clone https://github.com/<your-username>/htwg-practical-nlp.git
```

!!! info

    If you fork a repository, a copy of the repository will then be created in your user space.
    Besides the contributing aspect, you can also work independently on your assignments and push your changes to your remote repository.

!!! tip

    Forking is a very common practice in open source developlment.
    If you are new to open source development and have not forked a repository before, this may be a good learning opportunity for you! 🤓

## Syncing you Fork

During the semester, it is very likely that the course repository will be updated.
To pull those updates from the course repository (aka the `upstream`), you need to do the following:

```sh
git fetch upstream
git checkout main
git merge upstream/main
```

You can also sync your fork in the web UI or GitHub CLI.
You can find more details in the official [GitHub docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).

!!! info

    Syncing your fork only updates your local copy of the repository.
    To update your fork on GitHub.com, you must [push your changes](https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository).
