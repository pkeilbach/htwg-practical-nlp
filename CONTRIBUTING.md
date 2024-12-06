# Contributing Guide

In thise course, we follow an open source development approach, and highly encourage and appreciate any kind of contributions. 👐

This guide will help you to get the most out of this course and get you started with your first contribution! 🚀

**⚠️ Please wait with your contributions until we officially kick off our course in the first lecture. This is to give everybody a fair chance to contribute and find suitable issues.**

## How to contribute

There are many ways in which you can contribute to the course.

Anything that you think improves the course repository should be suitable for a contribution.

Every contribution is welcome - and some can be [rewarded](#rewarding-your-contributions)! 🏅

Here is some inspiration:

- Improve the documentation, setup guides, FAQs, etc. 📘
- Fix already known [bugs](https://github.com/pkeilbach/htwg-practical-nlp/labels/bug) 🐞
- Report unknown bugs (e.g. in the assignments) - and ideally try to fix them 🔧
- Work on [open issues](https://github.com/pkeilbach/htwg-practical-nlp/issues) (specifically watch out for the [`good first issue`](https://github.com/pkeilbach/htwg-practical-nlp/labels/good%20first%20issue) label) 🐣
- Engage in [discussions](https://github.com/pkeilbach/htwg-practical-nlp/discussions) 🗣️
- fix some typos 🖊️
- ...

Once you found something you can contribute:

- create a [new issue](https://github.com/pkeilbach/htwg-practical-nlp/issues/new/choose), or
- comment on an [existing issue](https://github.com/pkeilbach/htwg-practical-nlp/issues) that you want to work on it

so that it is clear to others and me that you are working on a topic.

> 💡 Note that you can always open a blank issue if no issue template is suitable.

After you implemented your changes, you need to open a [pull request](#making-contributions-to-the-course-repository) so that I can review your changes.

When I approve, we will merge your pull request, and besides being [rewarded](#rewarding-your-contributions), you will be listed as a [contributor](https://github.com/pkeilbach/htwg-practical-nlp/graphs/contributors)! 🎉

## Engage in discussions

You will get the most out of this course if you actively participate and exchange with both your peers and me as the lecturer.

We use [GitHub discussions](https://github.com/pkeilbach/htwg-practical-nlp/discussions) as a tool to discuss all things related to the course content.
Both your peers and me can comment on your topic and hopefully resolve your issue.

Here is how it works:

- Stuck with an assignment? Post your question to the [Q&A board 🙏](https://github.com/pkeilbach/htwg-practical-nlp/discussions/categories/q-a), maybe your peers have the same issue
- Anything that was covered in the lecture was confusing? Post it to the [Q&A board 🙏](https://github.com/pkeilbach/htwg-practical-nlp/discussions/categories/q-a) and I will get back to it
- Having a more general question? Use the [general board 💬](https://github.com/pkeilbach/htwg-practical-nlp/discussions/categories/general)
- You have an idea what could be improved? Use the [idea board 💡](https://github.com/pkeilbach/htwg-practical-nlp/discussions/categories/ideas)

Using the GitHub discussion tool is highly encouraged.
Every question is welcome and there is no such thing as stupid questions.
We all come from different backgrounds, and usually, if somehing is unclear to you, you are not alone. 👐

Also, initiating discussions offers the chance to earn bonus points, since a question of yours might reveal a bug or issue that needs to be solved! 🏅

> ℹ️ Note that "internal" announcements that are specific to the current semester will still be posted in Moodle, so that nobody misses anything.

## Making contributions to the course repository

If you have something that you can contribute to the course repository, you need to open a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

Since you are usually not listed as collaborator, you will need to [create a pull request from your fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

When creating your pull request, please consider the [best practices for creating pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/best-practices-for-pull-requests#best-practices-for-creating-pull-requests).

## Working with `pre-commit` hooks

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

A rewardable contribution is a **code change** that is successfully merged to the main branch of the course repository.

> While you could argue that engaging in discussions is also a kind of contribution, this is not rewardable contribution (but still encouraged 😃).
> However, a question of yours might reveal a bug or issue that needs to be solved! 🏅

You can make as many contributions as you want but you can earn a **maximum of 10 bonus points** throughout the semester.

> For example, if you reached 86 out of 100 points in the exam, and made contributions worth 6 bonus points, then your total points will be 92! 🚀
> Also note that completing [assignments](https://pkeilbach.github.io/htwg-practical-nlp/assignments/) can also earn you bonus points.

Each contribution should be a separate [pull request](#making-contributions-to-the-course-repository).

I will decide individually, based on the contribution, how much bonus points your contribution is worth.
But as a rule of thumb, you can assume that each contribution is worth 1 bonus point.

> For example, if you spot a bug in the assignments and provide the fix by yourself, this is probably worth 2-3 bonus points.
> On the other hand, fixing a single typo like in `orsnge` 🍊, is probably not worth a bonus point.
> However, fixing 5 typos with one pull request probably is worth a bonus point.
