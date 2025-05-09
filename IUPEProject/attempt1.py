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

#account	click	here	meeting	now	room	today	verify
# Email 0	0.25	0.25	0.25	0	0.25	0	0	0.25
# Email 1	0	0	0	0.33	0	0.33	0.33	0

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
