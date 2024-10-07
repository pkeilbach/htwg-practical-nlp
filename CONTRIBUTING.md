# Contributing Guide

In thise course, we follow an open source development approach, and highly encourage and appreciate any kind of contributions. 👐

This guide will help you to get the most out of this course and get you started with your first contribution! 🚀

## How to contribute

There are no strict guidelines about the type of contributions.
Anything that you think improves the course repository should be suitable for a contribution.

Every contribution is welcome - and will be [rewarded](#rewarding-your-contributions)! 🏅

Here is some inspiration:

- Improve the documentation and setup guides 📘
- Fix already known [bugs](https://github.com/pkeilbach/htwg-practical-nlp/labels/bug) 🐞
- Report unknown bugs (e.g. in the assignments) - and ideally try to fix them 🔧
- Work on open issues (specifically watch out for the [`good first issue`](https://github.com/pkeilbach/htwg-practical-nlp/labels/good%20first%20issue) label) 🐣
- fix some typos 🖊️
- ...

You can also engage in [discussions](https://github.com/pkeilbach/htwg-practical-nlp/discussions).
While this is not a contribution per se, a question of yours about an assignment or lecture may lead to an issue that needs to be solved!

Once you found something you can contribute, assign yourself to the issue so that it is clear to others that you are working on this topic.

After you implemented your changes, you need to open a [pull request](#making-contributions-to-the-course-repository) so that I can review your changes.

When I approve, we will merge your pull request, and besides being [rewarded](#rewarding-your-contributions), you will be listed as a [contributor](https://github.com/pkeilbach/htwg-practical-nlp/graphs/contributors)! 🎉

## Engage in discussions

You will get the most out of this course if you actively participate and exchange with both your peers and me as the lecturer.

We use [GitHub discussions](https://github.com/pkeilbach/htwg-practical-nlp/discussions) to discuss all things related to the course content.
Both your peers and me can comment on your topic and hopefully resolve your issue.

- Stuck with an assignment? Post your question to the [Q&A board 🙏](https://github.com/pkeilbach/htwg-practical-nlp/discussions/categories/q-a), maybe your peers have the same issue
- Anything that was covered in the lecture was confusing? Post it to the [Q&A board 🙏](https://github.com/pkeilbach/htwg-practical-nlp/discussions/categories/q-a) and I will get back to it
- Having a more general question? Use the [general board 💬](https://github.com/pkeilbach/htwg-practical-nlp/discussions/categories/general)
- You have an idea what could be improved? Use the [idea board 💡](https://github.com/pkeilbach/htwg-practical-nlp/discussions/categories/ideas)

Using the GitHub discussion tool is highly encouraged.
Every question is welcome and there is no such thing as stupid questions.
We all come from different backgrounds, and usually, if somehing is unclear to you, you are not alone. 👐

Also, initiating discussions offers the chance to earn bonus points, since a question of yours might reveal a bug or issue that needs to be solved! 🏅

> "Internal" announcements that are specific to the current semester will still be posted in Moodle, so that nobody misses anything.

## Forking the course repository

The recommended way to work with this course is to [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks) the course repository.

While cloning, as described in the [getting started guide](./docs/getting_started.md#clone-the-repository), lets you _participate_ in the course and work on the assignments, only forking the course repository allows you to _contribute_ back bug fixes, enhancements, etc.

To fork this repository, you can follow the official GitHub documentation on how to [fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

If you fork a repository, a copy of the repository will then be created in your user space.
Besides the contributing aspect, you can also work independently on your assignments and push your changes to your remote repository.

Cloning and setting up the environment works as described in the [getting started guide](./docs/getting_started.md#clone-the-repository), besides that you need to clone your fork to your local machine (instead of the course repository):

```sh
git clone https://github.com/<your-username>/htwg-practical-nlp.git
```

> 💡 Forking is a very common practice in open source developlment.
> If you are new to open source development and have not forked a repository before, this may be a good learning opportunity for you! 🤓

## Syncing your fork

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

## Development setup

Once you forked the course repository, you can start with the development setup.

The development setup is intented for contributors, and will install some additional dependencies (see the optional dependencies section in the [`pyproject.toml`](https://github.com/pkeilbach/htwg-practical-nlp/blob/main/pyproject.toml) file).
Specifically it will install some `pre-commit` hooks to assert that your contributed code meets some basic quality standards (see [below](#working-with-pre-commit-hooks)).

You can setup the development environment as follows:

```
make dev-setup
```

## Making contributions to the course repository

If you have something that you can contribute to the course repository, you need to open a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

Since you are usually not listed as collaborator, you will need to [create a pull request from your fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

When creating your pull request, please consider the [best practices for creating pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/best-practices-for-pull-requests#best-practices-for-creating-pull-requests).

## Working with pre-commit hooks

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

You can make as many contributions as you want but you can earn a **maximum of 5 bonus points**.

> For example, if you reached 86 out of 100 points in the exam, and made contributions worth 5 bonus points, then your total points will be 91 🚀

Each contribution should be a separate [pull request](#making-contributions-to-the-course-repository).

I will decide individually, based on the contribution, how much bonus points your contribution is worth.
But as a rule of thumb, you can assume that each contribution is worth 1 bonus point.

> For example, if you spot a bug in the assignments and provide the fix by yourself, this is probably worth 2-3 bonus points.
> On the other hand, fixing a single typo like in `orsnge` 🍊, is probably not worth a bonus point.
> However, fixing 5 typos with one pull request probably is.
