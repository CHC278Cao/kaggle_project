
import numpy as np
import pandas as pd
import re
import emoji

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def missing_data_percent(df):
    """
        Get the percentage of null values in all features
    :param df: Type: DataFrame
    :return: Type: DataFrame
    """
    total_num = df.isnull().sum().sort_values(ascending=False)
    percentage = round(total_num / len(df) * 100, 2)
    return pd.concat([total_num, percentage], axis=1, keys=["Total", "Percentage"])


def count_null_values(df, feature):
    """
        Get the percentage of null values in the special features
    :param df: Type: DataFrame
    :param feature: Type: str, the feature to be chosen
    :return: Type: DataFrame,
    """
    total_num = df.loc[:, feature].value_counts(dropna=False)
    percentage = round(df.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
    return pd.concat([total_num, percentage], axis=1, keys=["Total", "Percentage"])


def cal_duplicate_values(df):
    """

    :param df:
    :return:
    """
    dup = []
    columns = df.columns
    for c in df.columns:
        dup.append(sum(df[c].duplicated()))
    return pd.concat([pd.Series(columns), pd.Series(dup)], axis=1, keys=["Column", "Count"])


def get_remove_url_content(content, remove = False):
    url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    if remove:
        text = re.sub(url_pattern, '', str(content))
    else:
        text = re.findall(url_pattern, str(content))
    return "".join(text)



def get_emoji_content(content):
    emo_content = emoji.demojize(content)
    emoji_pattern = re.compile(r'\:(.*?)\:')
    text = re.findall(emoji_pattern, emo_content)
    return text

def remove_emoji_content(content):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', content)


def get_remove_email_content(content, remove=False):
    email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
    if remove:
        text = re.sub(email_pattern, '', str(content))
    else:
        text = re.findall(email_pattern, str(content))
    return "".join(text)


def get_hash_content(content):
    hash_pattern = re.compile(r'(?<=#)\w+')

    text = re.findall(hash_pattern, str(content))
    return " ".join(text)


def get_at_content(content):
    at_pattern = re.compile(r'(?<=@)\w+')

    text = re.findall(at_pattern, str(content))
    return " ".join(text)


def find_nonalp(content):
    """
        Retrieve all non alphanumber
    :param content: Type: str, content to be processed
    :return:
        a list containing all non alphanumber
    """
    text = re.findall("[^A-Za-z0-9 ]", str(content))
    return text


def find_punct(content):
    """
        Rrtrieve all punctuation
    :param text: Type: str, content to be processed
    :return:
        a list containing all punctation
    """
    text = re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', content)
    text = "".join(text)
    return list(text)


def find_stop_word(content):
    """
        retrieve all stop words
    :param content: Type: str, content to be processed
    :return:
        a list containing all stop words
    """
    stop_words = set(stopwords.words('enlish'))
    word_tokens = word_tokenize(content)
    non_stop_words = [w for w in word_tokens if w not in stop_words]
    stop_words = [w for w in word_tokens if w in stop_words]
    return stop_words


def get_only_words(content):
    """
        Retrieve sentence which only contains words
    :param content: content to be processed
    :return:
        Sentence which only contains words
    """
    re_pattern = re.compile(r'\b[^\d\W]+\b')
    text = re.findall(re_pattern, str(content))
    return " ".join(text)


def get_only_numbers(content):
    """
        Retrieve sentence which only contains numbers
    :param content: Type: str, content to be processed
    :return:
        Sentence which only contains numbers
    """
    re_pattern = re.compile(r'\b\d+\b')
    text = re.findall(re_pattern, str(content))
    return " ".join(text)


def get_only_key_sentence(content, keyword):
    """
         Retrieve sentence which only contains keywords
     :param content: Type str, content to be processed
     :param keyword: Type: str, keyword to be matched
     :return:
         Sentence which only contains keywords
     """
    re_pattern = re.compile(r'([^.]*'+keyword+'[^.]*)')
    text = re.findall(re_pattern, str(content))
    return text


def get_unique_sentence(content):
    """
          Retrieve sentence which only contains unique content
      :param content: content to be processed
      :return:
          Sentence which only contains unique content
      """
    re_pattern = re.compile(r'(?sm)(^[^\r\n]+$)(?!.*^\1$)')
    text = re.findall(re_pattern, str(content))
    return text



def ngrams_top(corpus, ngram_range, n=None):
    """
        Retrieve the top n ngrams
    :param corpus:  Type: pd.Series, corpus to be processed
    :param ngram_range: Type: tuple, range of gram numbers
    :param n: Type: int, the top n words
    :return:
        A dataframe which columns contrains words and word_frequency
    """
    vec = CountVectorizer(stop_words='english', ngram_range=ngram_range)
    bag_of_word = vec.fit_transform(corpus)
    sum_words = bag_of_word.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    total_list = words_freq[:n]
    df = pd.DataFrame(total_list, columns=['text', 'count'])
    return df



if __name__ == "__main__":
    # sentence = "I love spending time at https://www.kaggle.com/"
    # sentence = "I love ⚽ very much 😁"
    # sentence = "Its all about \U0001F600 face"
    # sentence = "My gmail is abc99@gmail.com"
    # sentence = "#Corona is #trending now"
    # sentence = "Corona virus have kiled #24506 confirmed cases now.#Corona is un(tolerable)"
    # sentence = "@David,can you @ help me out
    # sentence = "People are fighting with covid these days.Economy has fallen down.How will we survice covid"
    sentence = "I thank doctors\nDoctors are working very hard in this pandemic situation\nI thank doctors"
    print(get_unique_sentence(sentence))
    # print(get_only_words(sentence))

