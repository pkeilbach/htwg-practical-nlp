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
make assignment_01
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

## Jupyter Notebooks

Some of the assignments are accompanied by Jupyter notebooks.

See the [Getting Started](./getting_started.md) guide for instructions on how to start the Jupyter server.
