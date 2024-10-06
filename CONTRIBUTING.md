# Contributing Guide

In thise course, we follow an open source development approach, and highly encourage and appreciate any kind of contributions. 👐

For example, you can help by improving the documentation, setup guides, the lecture notes, provide bug fixes for the assignments, work on open issues (watch out for the [good first issue label](https://github.com/pkeilbach/htwg-practical-nlp/labels/good%20first%20issue)) or just fix some typos.
However small it is, every contribution is welcome - and will probably be rewarded (TBD)! 🏆

This guide will help you to get started with your first contribution! 🚀

## Forking the Course Repository

The recommended way to work with this course is to [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks) the course repository.

While cloning, as described in the [getting started guide](./docs/getting_started.md#clone-the-repository), lets you _participate_ in the course and work on the assignments, only forking the course repository allows you to contribute back bug fixes, enhancements, etc.

To fork this repository, you can follow the official GitHub documentation on how to [fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

If you fork a repository, a copy of the repository will then be created in your user space.
Besides the contributing aspect, you can also work independently on your assignments and push your changes to your remote repository.

Cloning and setting up the environment works as described in the [getting started guide](./docs/getting_started.md#clone-the-repository), besides that you need to clone your fork to your local machine (instead of the course repository):

```sh
git clone https://github.com/<your-username>/htwg-practical-nlp.git
```

> 💡 Forking is a very common practice in open source developlment.
> If you are new to open source development and have not forked a repository before, this may be a good learning opportunity for you! 🤓

## Syncing you Fork

During the semester, it is very likely that the course repository will be updated.
To pull those updates from the course repository (aka the `upstream`), you need to do the following:

```sh
git fetch upstream
git checkout main
git merge upstream/main
```

You can also sync your fork in the web UI or GitHub CLI.
Find more details in the official [GitHub docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).

> 💡 Syncing your fork only updates your local copy of the repository. To update your fork on GitHub.com, you must [push your changes](https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository).

## Development Setup

Once you forked the course repository, you can start with the development setup.

The development setup is intented for contributors, and will install some additional dependencies.

You can setup the development environment as follows:

```
make dev-setup
```

## Working with Pre-commit Hooks

One such additional dependency is the [`pre-commit` framework](https://pre-commit.com/).

On every commit, it runs some basic code quality checks to ensure that the contributed code satisfies certain standards, e.g. the [Black code formatter](https://black.readthedocs.io/en/stable/) for Python.

> 💡 You can check all currently implemented `pre-commit` hooks in the [`pre-commit-config.yaml`](https://github.com/pkeilbach/htwg-practical-nlp/blob/main/.pre-commit-config.yaml).

The hooks are executed on the staged files, and it may happen that the hooks make changes the staged files (e.g. formatting), so you need to stage them again before finally comitting them:

```sh
# stage your files
git add .

# commiting your files will trigger the pre-commit hooks
git commit -m "some cool updates"

# assuming that your files violated some formatting rules,
# the formatter hook will try to fix them, so you need to stage them again
git add .

# now the commit shall pass
git commit -m "some cool updates"
```

## Rewarding your contributions

To make contributions to the course repository more attractive for you, each contribution will be **rewarded with bonus points** that will be transferred to the exam. 🏅

You can make as many contributions as you want but you can earn a **maximum of 5 bonus points** (TBD).

> For example, if you reached 86 out of 100 points in the exam, and made contributions worth 5 bonus points, then your total points will be 91 🚀

Each contribution should be a separate [pull request](#making-contributions-to-the-course-repository).

I will decide individually, based on the contribution, how much bonus points your contribution is worth.
But as a rule of thumb, you can assume that each contribution is worth 1 bonus point.

> For example, if you spot a bug in the assignments and provide the fix by yourself, this is probably worth 2-3 bonus points.
> On the other hand, fixing a single typo like in `orsnge` 🍊, is probably not worth a bonus point.
> However, fixing 5 typos with one pull request probably is.
