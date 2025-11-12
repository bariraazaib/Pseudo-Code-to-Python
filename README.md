# Pseudo-Code-to-Python
🖥️ Pseudo-code to Python Generator

Ye project GPT-2 + LoRA model ka use karke pseudo-code ko executable Python code me convert karta hai. Streamlit aur Gradio dono supported hain.

🚀 Features

Pseudocode input → Python code output

Fine-tuned GPT-2 model with LoRA for faster training

Real-time generation via Streamlit UI

Tested on multiple examples like loops, sum, max calculation

📦 Repo Structure
repo/
│
├── app.py                  # Streamlit app
├── requirements.txt        # Dependencies
└── pseudo-to-python-final/ # Fine-tuned GPT-2 + LoRA model

⚡ Quick Start

Clone repo:

git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>


Install dependencies:

pip install -r requirements.txt


Run Streamlit app:

streamlit run app.py


Open browser → enter pseudocode → get Python code 🐍

💡 Examples
Pseudocode	Generated Python Code
print numbers from 1 to 10	for i in range(1, 11): print(i)
sum of two numbers	a = int(input()); b = int(input()); print(a + b)
find maximum in list	lst = [int(x) for x in input().split()]; print(max(lst))
📌 Notes

Model requires GPU for faster inference (CUDA recommended)

You can also use Gradio interface if preferred

Make sure pseudo-to-python-final/ folder is in repo
