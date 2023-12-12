# Presentations

During the semester, each student should give a **presentation about an NLP tool**, frameork, or related topic.
The idea is that you can explore an exciting NLP technology, play around with it and share your findings with the class.

Maybe you can think of it as if you were part of a software engineering team working on an NLP project, and your task is to evaluate if a specific technology is suitable for the project and share your findings with the team.
After your presentation, every team member should have a good understanding of the technology and its capabilities.

There are two parts to the presentation:

1. A **verbal presentation** in the lecture, including a **demo or practical example**
2. A **blog article** about the topic

Both of them are ungraded, but they are mandatory to pass the course.

!!! note

    Since this is an NLP course, it is totally acceptable to use **LLMs or GenAI** to generate the content for your presentation. If you do so, please share your experiences at the end of the presentation! 🤖

## Verbal Presentation

The scope of the verbal presentation should be around **15-20 minutes** and you should include a **demo or practical example**. 🧑‍🏫

For the **demo**, try to aim for 30-50% of the time (depending on the topic). Feel free to use any tool you want. You can show something in the browser, use Postman or Jupyter notebooks, or do some live hacking on the command line. 🤓

Besides that, there are no strict rules about the presentation format. You can use slides if you want, but it is not mandatory. 📊

!!! info "Submission"

    You **don't need to submit** any presentation **slides** or other assets from your verbal presentation.

    You only need to submit the **blog article** (see below).

## Blog Article

In addition to the verbal presentation, the student should submit a **blog-like article** about the topic based on this [Markdown template](https://github.com/pkeilbach/htwg-practical-nlp/blob/main/docs/presentations/articles/template.md). 📝

Keep the article short and crisp. There are no hard rules, but maybe a **reading time** of not more than **10 minutes** is a good approximation. ⏳

The article should be written in **English** and should be **self-contained**. :gb:

You have the following options to **submit the article**: 📬

1.  **Via pull request to the Github repository (preferred) 🤓**

    Please add your article to the `docs/presentations/articles/` folder and name it `<your topic>.md`.

    Then create a pull request to the `main` branch of the repository.

    ```sh
    git checkout -b presentation-<your topic>
    git add docs/presentations/articles/<your topic>.md
    git commit -m "Add blog article for <your topic>"
    git push -u origin presentation-<your topic>
    ```

    Through [GitHub](https://github.com/pkeilbach/htwg-practical-nlp/pulls), you can then create a pull request to the `main` branch.

    Please add the `presentation topic` label to your pull request, and add me as a reviewer.

    !!! note "Pull Requests ✅"

        If you are not familiar with GitHub and pull requests, this is a great opportunity to learn about it!
        It is a very common workflow in software development, and you will probably encounter it in your future career.

        You can find a quick guide [here](https://guides.github.com/activities/hello-world/), and learn more about pull requests [here](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) and [here](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

    !!! tip "Shortcut 💫"

        Usually, after pushing your branch to the remote repository, GitHub will already **suggest** you to create a pull request via a "Compare & pull request" banner.

    !!! warning "Publishing 📣"

        If you submit via pull request, you implicitly agree to publish the article on the course website. However, feel free to remove the author information if you don't want to be named.

2.  **Via email to the lecturer 📧**

    You can just send me the markdown file via [email](mailto:pascal.keilbach@htwg-konstanz.de).
    Please let me know if it is OK to publish the article on the course website.

!!! info "Submission Deadline 📅"

    The article does not need to be submitted on the same day as the presentation. You can submit it later, but please make sure to submit it before the **end of the semester** (the exact date will be announced).

## Topics and Dates

You can find the **list of topics** in the GitHub issues when filtering for the [`presentation topic` label](https://github.com/pkeilbach/htwg-practical-nlp/issues?q=is%3Aissue+is%3Aopen+label%3A%22presentation+topic%22).

If you are interested in a topic, please add yourself as an **assignee** to the issue.

You can also suggest your **own topic**. If you do so, please create a new issue and add the `presentation topic` label.

If multiple students are interested in the same topic, we will figure out a way to assign the topics fairly.
In case of doubt, a first-come-first-serve approach may be applied, so it's probably a good idea to be quick! 😉

As for the **dates**, we will discuss them in one of the first lectures.
