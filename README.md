Based on the contents of the [BayesianInferenceUB25 GitHub repository](https://github.com/enriquemoraayala/BayesianInferenceUB25), here is a well-structured and enhanced `README.md` file:

```markdown
# BayesianInferenceUB25

📊 Bayesian Inference Course  
🎓 MSc in Data Science — University of Barcelona (UB)  
👥 Instructors: Nahuel Statuto & Enrique Mora

---

## 🧠 About the Course

This repository hosts the course materials for the **Bayesian Inference** subject in the Master's in Data Science program at the **University of Barcelona**.

Students will gain theoretical and practical foundations in Bayesian statistics, learning how to:
- Model uncertainty using probability
- Build and interpret Bayesian models
- Perform inference using tools like **PyMC**
- Evaluate model performance and communicate results

---

## 📂 Repository Structure

```
├── notebooks/      # Jupyter notebooks with practical examples and exercises
├── slides/         # Lecture slides (PDF format)
├── LICENSE         # License (Apache 2.0)
└── README.md       # This file
```

---

## 🚀 Getting Started

To use the materials locally:

### 1. Clone the repository
```bash
git clone https://github.com/enriquemoraayala/BayesianInferenceUB25.git
cd BayesianInferenceUB25
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\\Scripts\\activate
```

### 3. Install dependencies
If a `requirements.txt` file is provided:
```bash
pip install -r requirements.txt
```

If not, install key packages manually:
```bash
pip install pymc numpy pandas matplotlib jupyter arviz
```

### 4. Launch Jupyter
```bash
jupyter notebook
```

Open the `notebooks/` directory and start exploring the content.

---

## 📘 Course Topics

- Bayesian probability and inference
- Conjugate priors and analytical solutions
- Markov Chain Monte Carlo (MCMC)
- Hierarchical models
- Probabilistic programming with PyMC
- Posterior predictive checks and diagnostics

---

## 🛠 Tools Used

- [PyMC](https://www.pymc.io/)
- [ArviZ](https://www.arviz.org/)
- [Jupyter Notebooks](https://jupyter.org/)
- Python libraries: NumPy, Pandas, Matplotlib

---

## 📄 License

This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

- **Nahuel Statuto**: [nstatuto@ub.edu](mailto:nstatuto@ub.edu)  
- **Enrique Mora**: [enriquemora@ub.edu](mailto:enriquemora@ub.edu)

Feel free to open issues or reach out for questions or collaboration opportunities!
```

Let me know if you'd like a `requirements.txt` or badges (e.g. binder, license, build status) added to the README!