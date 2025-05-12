import pandas as pd #dataframe
import string

#Dataframe
df = pd.read_csv('Phishing_Email.csv')
df_subset = df.head(1000)

#dataset columns: "Index(['Unnamed: 0', 'Email Text', 'Email Type'], dtype='object')"
df_subset = df_subset.drop(columns=['Unnamed: 0']) #unnamed column isnt very useful

#rename columns to simpler names
df_subset = df_subset.rename(columns={
    'Email Text': 'text',
    'Email Type': 'label'
})

def clean_text(text):
    text = str(text).lower() 
    text = "".join(char for char in text if char not in string.punctuation)  
    return text

df_subset['cleaned'] = df_subset['text'].apply(clean_text) #create a new column called cleaned

print(df_subset[['text', 'cleaned']].head())

#vectorize the text
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer() 

X = vectorizer.fit_transform(df_subset['cleaned']) #each row is email, each column is word
y = df_subset['label']

#train regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_indices = np.argsort(coefficients)[-10:]
bottom_indices = np.argsort(coefficients)[:10]

'''

print("Top 10 phishing words:")
for i in reversed(top_indices):
    print(f"{feature_names[i]}: {coefficients[i]:.3f}")

print("\nTop 10 safe words:")
for i in bottom_indices:
    print(f"{feature_names[i]}: {coefficients[i]:.3f}")
    
'''   

'''
def predict_email(email_text):
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

while True:
    user_input = input("\nPaste an email to check (or type 'exit' to quit):\n")
    if user_input.lower() == "exit":
        break
    result = predict_email(user_input)
    print(f"\nThis is a **{result.upper()}**.\n")

'''   


#------------------------
#User Interface
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def predict_email(email_text):
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

def on_check():
    user_text = input_box.get("1.0", tk.END).strip()
    if not user_text:
        messagebox.showwarning("Input Needed", "Please paste an email.")
        return
    result = predict_email(user_text)
    output_label.config(text=f"Prediction: {result.upper()}")

def toggle_view():
    if input_box.winfo_viewable():
        #hide detector, show info
        input_box.pack_forget()
        check_button.pack_forget()
        output_label.pack_forget()
        image_label.pack_forget()
        image_label2.pack()
        info_label.pack(pady=10)
        toggle_button.config(text="Back")
    else:
        #show detector, hide info
        info_label.pack_forget()
        input_box.pack(pady=10)
        check_button.pack(pady=10)
        output_label.pack(pady=10)
        image_label.pack()
        image_label2.pack_forget()
        toggle_button.config(text="Show Info")

root = tk.Tk()
root.title("Phisher Diminisher")
root.geometry("800x500")

title = tk.Label(root, text="Phisher Diminisher", font=("Helvetica", 16))
title.pack(pady=10)

toggle_button = tk.Button(root, text="Show Info", command=toggle_view)
toggle_button.pack(pady=5)

input_box = tk.Text(root, height=10, width=60)
input_box.pack(pady=10)

check_button = tk.Button(root, text="Check Email", command=on_check)
check_button.pack(pady=10)

output_label = tk.Label(root, text="", font=("Helvetica", 14), fg="blue")
output_label.pack(pady=10)

try:
    image = Image.open("walter.png")
    image = image.resize((400, 400))
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=photo)
    image_label.pack()
    
    image2 = Image.open("tralalero.png")
    image2 = image2.resize((400,400))
    photo2 = ImageTk.PhotoImage(image2)
    image_label2 = tk.Label(root, image=photo2)
    image_label2.pack()
except:
    image_label = tk.Label(root, text="[Image Not Found]")
    image_label.pack()
    
    

info_text = (
    "Phishing is a type of cyberattack where attackers impersonate trustworthy entities "
    "via email or messaging to trick individuals into providing sensitive data like passwords, "
    "bank details, or personal information. These messages often contain malicious links or attachments."
)
info_label = tk.Label(root, text=info_text, wraplength=700, justify="left", font=("Helvetica", 40), bg="lightyellow")

root.mainloop()