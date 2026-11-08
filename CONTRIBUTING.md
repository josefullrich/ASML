# Team workflow

Each teammate clones the shared repository to their own computer, creates their
own Python environment using `README.md`, and works on a separate branch.

Before starting a change:

```bash
git switch main
git pull --ff-only
git switch -c your-name/describe-change
```

Edit and check your changes. Stage specific files, inspect the diff, then commit
and push:

```bash
git add src/example_bayesopt.py
git diff --cached
git commit -m "Describe the change"
git push -u origin your-name/describe-change
```

Open a pull request on GitHub from your branch into `main`. Describe the change
and how you checked it. Ask a teammate to review it, address feedback, and merge
when ready. Then everyone updates their local `main` with `git pull --ff-only`.
If you have unfinished local changes, commit them on your working branch before
switching branches. Resolve any conflicts deliberately; avoid force-pushing
shared branches.

Keep reusable logic in `src/` and experiment configuration/explanation in
`notebooks/`. Clear notebook outputs before committing and save executed copies
under `results/`. Avoid unrelated notebook metadata changes in pull requests.

`data/`, `docs/`, `results/`, and `.venv/` are excluded from Git. Share original
datasets and project PDFs separately using the team's agreed location. Do not
commit credentials or access tokens.

For Colab, upload the current `src/` folder or clone the repository into the
runtime using your own authenticated access. Keep authentication outside the
notebook's saved code and outputs. Local edits do not automatically update the
runtime. Record the branch/commit and configuration used for a run, and save
results to Drive or download them before removing the runtime.

The current optimizer uses a soft die-yield penalty. Do not describe it as
enforcing a hard 99.5% yield constraint until that change has been implemented
and checked.
