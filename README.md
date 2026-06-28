<h1 align="center">🛡️ AdverShield</h1>
<h3 align="center">Adversarial ML Battle Arena</h3>

<p align="center">
  A full-stack platform where AI agents compete to attack and defend image classifiers — bringing adversarial machine learning to life through an interactive battle arena.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&amp;logo=python&amp;logoColor=white"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&amp;logo=tensorflow&amp;logoColor=white"/>
  <img src="https://img.shields.io/badge/Groq%20API-F55036?style=for-the-badge&amp;logo=meta&amp;logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/>
</p>

---

## 📖 Overview

**AdverShield** simulates a battle between two AI agents — an **attacker** and a **defender** — competing over image classification models. The attacker tries to fool the classifier using adversarial perturbations, while the defender applies countermeasures to maintain accuracy. The project combines classic adversarial ML techniques with modern LLM-driven strategy generation and reinforcement learning, wrapped in a real-time analytics dashboard.

---

## ✨ Key Features

- ⚔️ **12+ Attack Algorithms** — including FGSM (Fast Gradient Sign Method), PGD (Projected Gradient Descent), and CW (Carlini-Wagner)
- 🛡️ **13 Defense Mechanisms** — implemented via the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- 🤖 **LLM-Powered Strategy Generation** — integrated **Llama 3.3** via the **Groq API** to generate adaptive attack/defense strategies
- 🧠 **Q-Learning Agent** — enables adaptive, reinforcement-learning-driven attack-defense behavior over multiple rounds
- 🔍 **GradCAM Visualizations** — highlights which regions of an image influence the classifier's decisions, before and after adversarial perturbation
- 📊 **Real-Time Analytics Dashboard** — tracks attack success rates, defense effectiveness, and model accuracy live as battles unfold

---

## 🏗️ How It Works

1. An image is fed into a target image classifier.
2. The **attacker agent** selects (or is guided by Llama 3.3 to select) an adversarial attack algorithm to perturb the image.
3. The **defender agent** applies one of 13 ART-based defense mechanisms to counter the perturbation.
4. The Q-Learning module updates each agent's strategy based on the round's outcome.
5. GradCAM visualizations and live metrics are pushed to the dashboard so the battle can be observed in real time.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Adversarial ML | ART (Adversarial Robustness Toolbox) |
| LLM Integration | Llama 3.3 via Groq API |
| Reinforcement Learning | Q-Learning |
| Visualization | GradCAM |
| Dashboard | Real-time analytics interface |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip
- A Groq API key (for Llama 3.3 integration)

### Installation

```bash
git clone https://github.com/mhammad-code/AdverShield-Adversarial-ML-Battle-Arena.git
cd AdverShield-Adversarial-ML-Battle-Arena
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the project root and add your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

### Run the project
```bash
python main.py
```

---

## 📸 Screenshots

> Add screenshots or a short demo GIF of the dashboard here once available.

---

## 📌 Future Improvements

- Add more attack/defense algorithm combinations
- Support for additional model architectures beyond image classifiers
- Multiplayer mode for human-guided attack/defense strategies
- Export battle history as detailed reports

---

## 👤 Author

**Muhammad Hammad Hussain**
📧 mhammadhussain81@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/hammad-h-b0ab66282) · [GitHub](https://github.com/mhammad-code)

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute with attribution.
