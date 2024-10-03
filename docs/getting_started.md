# Getting Started

This page describes how to set up your environment for this course.

## Accounts

To get the most out of this course, you should have a [GitHub](https://github.com/) and [Mural](https://www.mural.co/) account.
Both services are free to use.

!!! tip

    You can use your HTWG email address to register for GitHub and Mural.
    This will make it easier to identify you as a member of this course.

## Install Python

The recommended Python version for this course is 3.12. in a virtual environment.

=== ":fontawesome-brands-linux: Linux"

    ```sh
    sudo apt update
    sudo apt install python3.12
    sudo apt install python3.12-venv
    ```

    In case this doesn't work, try to add the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) to your system, and try again.

    ```sh
    sudo add-apt-repository ppa:deadsnakes/ppa
    ```

=== ":fontawesome-brands-apple: Mac"

    On Mac, you can use [Homebrew](https://brew.sh/) to install Python.

    ```sh
    brew install python@3.12
    ```

=== ":fontawesome-brands-windows: Windows"

    On Windows, it is recommended to use the [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/).
    Then you can follow the instructions for Linux.

    If you are a VS Code user, you need to [install the WSL extension](https://code.visualstudio.com/docs/remote/wsl).

    There is currently no setup guide for native Windows, but I'm happy to accept a pull request for [this issue](https://github.com/pkeilbach/htwg-practical-nlp/issues/12). 😉

!!! warning

    You are free to use another Python version if you wish, but be aware that this may cause problems with the provided code.
    Also if you are using Python outside a virtual environment or with a distribution like Anaconda, the described setup may not work.

## Fork and clone the repository

Make sure you have [Git](https://git-scm.com/) installed on your system.

Then, create a fork using the GitHub WebUI, and clone the repository to your local machine.

<!-- TODO issue-123 reference contribution guide-->

!!! info

    If you are new to this, you can follow the official GitHub documentation on how to [fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

!!! note

    You can also clone the course repository directly, but then you cannot contribute back bug fixes or enhancements (which would be highly appreciated! 👐). You can find more details about this in the contributing guide.

## Execute the Setup Script

The setup script is provided as a `Makefile`.
Change into the repository directory and execute the setup script.
This should create a virtual environment and install all required dependencies.

```sh
cd htwg-practical-nlp
make
```

This may take a few minutes. ☕

If everything went well, you should be good to go.

## Test your Installation

You can test your installation by running the tests for the first assignment.

```sh
make assignment_1
```

In your terminal, you should see lots of failed tests. 😨

But this is exactly what we want to see, since we haven't implemented anything yet! 🤓

!!! info

    You can find more details on how we handle assignments on the [corresponding page](./assignments.md).

## Jupyter

Some of the assignments are accompanied by Jupyter notebooks.

If your IDE supports it, you can execute the Jupyter notebooks natively in your IDE (e.g. using the [VSCode Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)).

IF you prefer the WebUI, you can start the Jupyter server with the following command.

```sh
make jupyter
```

Jupyter is now accessible at <http://localhost:8888/>.

!!! info

    Of course you can also use JupyterLab if you wish, but this is not included in the setup script.

## Serve the Lecture Notes

If you want, you can bring up the lecture notes on your local machine.

```sh
make docs
```

The lecture notes are now accessible at <http://localhost:8000/>.

---

If you came this far, your initial setup was successful and you are ready to go! 🚀
