import re
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+") #delete urls
MENTION_RE = re.compile(r"@\w+") #delete mentions
HASHTAG_RE = re.compile(r"#+") #delete hashtags
SPECIAL_RE = re.compile(r"[^a-z0-9\s]") #delete special characters
WHITESPACE_RE = re.compile(r"\s+") #replace multiple whitespace with single space

EMOJI_RE = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
) #delete emojis

def normalize_elongated_text(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def clean_text(text):
    if text is None:
        return ""
    
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = normalize_elongated_text(text)
    text = SPECIAL_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)     
    
    return text.strip()


# example to check it's working
# print(clean_text("This is    an example!!! Visit http://example.com @user ##hashtag 😊😊😊")) 


def main():
    df = pd.read_csv("../data/Tweet Classification.csv")

    df = df.drop(columns=["keyword", "location"], errors="ignore") #drop unwanted columns
    df["text"] = df["text"].apply(clean_text) #replace text column with cleaned text

    df.to_csv("../data/cleaned_data.csv", index=False) #save everything else unchanged

    print("Data cleaning completed. Text column replaced with cleaned text.")

    
    
if __name__ == "__main__":
    main()

