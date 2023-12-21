# Assignments

During the semester, each student has to complete a couple of assignments.
The assignments are ungraded, but they are mandatory to pass the course.

## Assignment Structure

Throughout the course, we will work on a Python package called `htwgnlp`.
The package is located in the `src` directory and is a fully functional and installable Python package.
The core sturcture will be provided, and the assignments will be about implementing the missing functionality.

To work on an assignment, you will need to locate the `TODO ASSIGNMENT-*` items in the code.
For example, to work on the first assignment, use the search functionality of your IDE to find all relevant items:

```txt
TODO ASSIGNMENT-1
```

!!! tip

    You should check the unit tests located in the `tests` directory to see the exact requirements that need to be implemented.

## Tests

Once you implemented everything, you can run the tests to check if everything works as expected.

You can run the tests using the `make` commands, for example:

```sh
make assignment_1
```

If all your tests pass, you successfully completed the assignment! 🚀

!!! tip

    If your IDE provides the functionality, you can also run the tests directly from the IDE.

!!! note

    You can also use the native `pytest` commands, but then you need to know the exact path to the tests:

    ```sh
    # make sure to have the virtual environment activated
    pytest tests/htwgnlp/test_preprocessing.py
    ```

!!! info

    Pytest is a very powerful testing framework and the de-facto standard for testing in Python.
    You will not need to know all the details, but if you want to learn more, check out the [official documentation](https://docs.pytest.org/en/latest/contents.html).

## Submitting Assignments

To submit an assignment, you will need to demonstrate a successful test run.

Also we may walk through your code together to check if you understood the most important concepts.

## Jupyter Notebooks

Some of the assignments are accompanied by Jupyter notebooks.

See the [Getting Started](./getting_started.md) guide for instructions on how to start the Jupyter server.

## Working on your Assignments locally

There are a few strategies on how to work on your assignments locally.

What you end up using is up to your personal preference.

### Using a single branch

This is probably the easiest way to work on your assignments and **generally the recommended way**.

You can create a new branch for your assignments:

```sh
git checkout -b my-assigments
```

Then, you can work on your assignments and commit your changes locally:

```sh
git add .
git commit -m "solution for assignment 1"
```

Whenever updates are available, you can switch back to the `main` branch, pull the latest changes, and merge them into your branch:

```sh
git checkout main
git pull
git checkout my-assigments
git merge main
```

!!! warning

    You should **not push** your branch to the remote repository.
    It should not be visible to others.
    Your implementation is only valid for yourself, in your local repository, and will not be merged into the `main` branch.

### Using multiple branches

Similarly, you could create a new branch for each assignment:

```sh
git checkout -b my-assigments-1
git add .
git commit -m "solution for assignment 1"
```

But note that you would need to merge updates from the `main` branch into each of your assignment branches.

Also keep in mind that some assignments build upon each other.

### Using stashing

You could also work on your assignments directly in the `main` branch, and whenever you need to pull the latest changes, you can stash your changes, pull the latest changes, and then pop your changes back:

```sh
git stash
git pull
git stash pop
```

Note that this way, you may lose your changes if you do not stash them properly.

See the [git stash documentation](https://git-scm.com/docs/git-stash) for more details.
