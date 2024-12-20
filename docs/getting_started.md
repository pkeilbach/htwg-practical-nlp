# Development Setup

This page describes how to set up your development environment for this course.

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

## Forking the Repository (Optional)

You can decide between forking the course repository, or just clone it:

- When [forking](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks) the repository, you act as a _conributor_ to the course repository. You will go through the full development setup and can use your remote repository to manage your work. Also, you can contribute back changes to earn bonus points for the exam. 🏅
- When [cloning](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the repository, you act as a _user_ of the course repository. While this is a bit more leightweight, you work fully locally and cannot contribute to the course repository. 🙁

!!! tip

    Forking is a very common practice in open source developlment.
    If you are new to open source development and have not forked a repository before, this may be a good learning opportunity for you! 🤓

If you decide to clone the repository, you can directly continue with [the next step](#clone-the-repository).

If you decide to fork the repository, you can follow the official GitHub documentation on how to [fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo).

!!! info

    If you fork a repository, a copy of the repository will be created in your personal GitHub user space.

## Clone the Repository

Make sure you have [Git](https://git-scm.com/) installed on your system.

Cloning the repository is straightforward, no matter if you work on a fork or not. You only need to watch out where you clone from:

```sh
# When cloning your fork, make sure to clone it from your personal GitHub user space
git clone https://github.com/<your-username>/htwg-practical-nlp.git

# cloning the course repository directly
git clone https://github.com/pkeilbach/htwg-practical-nlp.git
```

## Execute the Setup Script

The setup script is provided as a `Makefile`.
Change into the repository directory and execute the setup script.
This should create a virtual environment and install all required dependencies.

```sh
# go to the project directory
cd htwg-practical-nlp

# use plain make to install all required dependencies
make

# if you plan to contribute, you need to install the dev dependencies
make install-dev
```

This may take a few minutes. ☕

If everything went well, you should be good to go.

!!! info "Acticate the virtual environment"

    From now, make sure that you have the virtual environment activated.
    Usually, the IDE should automatically suggest you to activate it (e.g. VSCode).
    If that is not the case, you can activate the virtual environment with the following command

    ```sh
    # activate the virtual environment manually
    source .venv/bin/activate

    # in case you need to deactivate it
    deactivate
    ```

## Test your Installation

You can test your installation by running the tests for the first assignment.

```sh
make assignment-1
```

In your terminal, you should see lots of failed tests. 😨

But this is exactly what we want to see, since we haven't implemented anything yet! 🤓

!!! info

    You can find more details on how we handle assignments in the [assignments guide](./assignments.md).

If you came this far, your initial setup was successful and you are ready to go! 🚀

Now we can take a look at some other components of the repository.

## Jupyter

Some of the assignments are accompanied by Jupyter notebooks.

If your IDE supports it, you can execute the Jupyter notebooks natively in your IDE (e.g. using the [VSCode Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)).

If you prefer the web UI, you can start the Jupyter lab server with the following command.

```sh
make jupyter
```

Jupyter is now accessible at <http://localhost:8888/>.

## Serve the Lecture Notes

If you want, you can bring up the lecture notes on your local machine.

```sh
make mkdocs
```

The lecture notes are now accessible at <http://localhost:8000/>.

## Fetching Updates

During the semester, it is very likely that the course repository will be updated.

You can incorporate the updates as follows:

```sh
# if you work on a fork, you need to fetch your updates from the course repository (aka 'upstream')
git fetch upstream
git checkout main
git merge upstream/main

# make sure the correct upstream repository is set:
git remote -v
# if not, add the upstream:
git remote add upstream https://github.com/pkeilbach/htwg-practical-nlp.git

# otherwise (if you have cloned the repository), you need to fetch from origin
git fetch origin
git checkout main
git merge origin/main
```

!!! info "Pull updates regularly"

    It is good practice to pull the latest changes from `main` every now and then (just in case you are wondering why your assignment tests suddenly fail 😅).
    However, important updates will be announced in the lecture.

!!! note

    Find more details about syncing a fork in the official [GitHub docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).
