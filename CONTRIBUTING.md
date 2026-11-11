# Contributing

Thank you for your interest! This kit is for teaching and learning with local language models, and it improves when teachers, researchers and students work on it together.

## Easy ways to help

- **Report how it went in your class or lab** (use the "Classroom report" issue template).
- **Report a bug** (use the "Bug report" template). Please include your system, the model you used and the exact message you saw.
- **Test on other hardware or models** and tell us the results.
- **Share ideas**: new exercises, languages, glossaries, evaluation sets, better methods.

## Sending a change (Pull Request)

1. Fork the repository and create a branch with a short name.
2. Keep the change small and focused. One idea per pull request is easiest to review.
3. Keep the style of the project: standard-library Python only (no `pip install`), English comments and messages, short and simple instructions, and user-selectable input and output paths.
4. Run `python -m py_compile` on the Python files you changed, and try your change with a real model if you can. Say in the pull request what you tested.
5. Update the README if you change how something is used.

## Please do not upload

- Model files or the KoboldCPP program (link to them instead).
- Copyrighted texts, or third-party glossaries that may not be redistributed (for example the Microsoft terminology files).
- Personal data, such as student names or local file paths.

By contributing, you agree that your contribution is released under the license of this repository.

## Disclaimer

Designed and architected for university translation technology labs using AI-assisted development; written purely against the Python standard library so every line remains inspectable for teaching and research.
