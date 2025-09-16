# BerNet

**BerNet** is a PyTorch extension for physics-informed neural networks (PINNs). Inspired by **Bernstein polynomials**, BerNet introduces an architectural design that enforce Dirchlet and Neumann boundary conditions.

## Repository

```bash
|- .vscode/: Hidden documents inside vscode.
|- docs/: Documentation files.
|- experiments/: Configurations, scripts, and logs for running and reproducing experiments.
|- notebooks/: Jupyter notebooks for exploratory analysis.
|- src/bernet/: Source code of BerNet.
|- tests/: Unit and integration tests to ensure correctness of the codebase.
|- tutorials/: Step-by-step guides and example workflows for new users.
|- .gitignore
|- LICENSE
|- pyptoject.toml
|- README.md
|- requirements.txt
```

## 👩‍💻 Contributing to BerNet

Follow these steps to set up your development environment and contribute code to BerNet.

---

### 1. Clone the repository
First, download the source code from GitHub:
```bash
git clone git@github.com:luralves/BerNet.git
cd BerNet
````

---

### 2. Create and activate a virtual environment

It’s recommended to work in a virtual environment so dependencies stay isolated.

**Create the environment:**

```bash
python3 -m venv .venv
```

**Activate it:**

* On **Linux/macOS**:

  ```bash
  source .venv/bin/activate
  ```
* On **Windows (PowerShell)**:

  ```powershell
  .venv\Scripts\Activate.ps1
  ```

When activated, your shell prompt will show `(.venv)`.

*(When you’re done working, exit with `deactivate`.)*

---

### 3. Install BerNet in development mode

With the virtual environment active, install the project and development dependencies:

```bash
pip install -e .[dev]
```

* `-e` (editable mode) makes code changes in `src/bernet/` immediately available.
* `[dev]` installs extra tools like `pytest` for testing.

---

### 4. Keep `requirements.txt` clean

If you add a new dependency, update `requirements.txt` with **only top-level dependencies**:

```bash
pip list --not-required --format=freeze > requirements.txt
```

This ensures we don’t commit all transitive packages (like `nvidia-*`).

---

### 5. Create a new branch

Never commit directly to `main`. Make sure your local repo is up to date:

```bash
git checkout main
git pull origin main
```

Then create a descriptive branch:

```bash
git checkout -b feature/my-feature
```

Examples:

* `feature/new-loss-function`
* `fix/docs-typo`

---

### 6. Make changes and commit

Edit the code, then stage and commit:

```bash
git add .
git commit -m "Add new loss function with per-point weights"
```

---

### 7. Push your branch

Send your branch to GitHub:

```bash
git push origin feature/my-feature
```

---

### 8. Open a Pull Request (PR)

1. Go to your fork or the main repository on GitHub.
2. GitHub will suggest opening a PR for your branch.
3. Provide a clear description of your changes.
4. Submit the PR and request a review.
