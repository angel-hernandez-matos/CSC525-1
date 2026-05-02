# File: mainPP.py
# Written by: Angel Hernandez
# Description: Module 8 - Portfolio Project - Option 2
# Requirement(s): Your chatbot must meet the following conditions:
#                 - Use NLP learning methods to respond to user inputs.
#                 - Chatbot responses should not be nonsense. Responses should be clearly in
#                   response to the input text.

import os
import sys
import json
import random
import subprocess
from enum import Enum

class DatasetType(Enum):
    NPS = 1
    CORNELL = 2
    WELLNESS = 3

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class AdaptiveControlMessages:
    FALLBACK_NEG = [
        "It sounds really heavy. I'm here with you — want to share a bit more?",
        "That seems tough. You don’t have to go through it alone.",
        "Your feelings matter. What part is weighing on you the most?"
    ]
    FALLBACK_NEU = [
        "I’m here to help. What’s on your mind?",
        "Thanks for reaching out. Tell me more so I can support you.",
        "Let’s take it one step at a time — what’s happening?",
        "I’m listening. What part would you like to talk through?",
        "You’re doing the right thing by checking in with yourself.",
        "We can sort through this together. What’s the first thing on your mind?",
        "I’m here with you. What feels unclear right now?",
        "It’s okay to pause and reflect. What’s coming up for you?",
        "Let’s explore this gently. What’s the situation?",
        "I’m here to support you. What would you like to focus on?",
        "You don’t have to rush. What’s the next detail you want to share?"
    ]
    FALLBACK_POS = [
        "I love that energy — want a tip to keep the momentum going?",
        "That’s great to hear! Want to build on that?",
        "Sounds good! What would you like to focus on next?"
    ]
    MOVIE_STYLE_PHRASES = [
        "what do you want me to say",
        "i don't know what you're talking about",
        "that's not my problem",
        "why would you ask me that"
    ]

class ChatbotEngine:
    def __init__(self):
        import spacy as sp
        import nltk as nltk
        from textblob import TextBlob as tb
        from chatterbot import ChatBot as cb
        from nltk.corpus import stopwords as sw
        from nltk.stem import WordNetLemmatizer as wl
        from chatterbot.trainers import ListTrainer as lt
        from sklearn.metrics.pairwise import cosine_similarity as cs
        from sklearn.feature_extraction.text import TfidfVectorizer as tv

        self.bot = None
        self.spacy = sp
        self.nltk = nltk
        self.ChatBot = cb
        self.TextBlob = tb
        self.corpus = None
        self.stopwords = sw
        self.ListTrainer = lt
        self.TfidfVectorizer = tv
        self.WordNetLemmatizer = wl
        self.cosine_similarity = cs

        # For hybrid similarity
        self.vectorizer = None
        self.tfidf_matrix = None
        self.raw_inputs = []
        self.raw_responses = []
        self.last_fallback = None

        self.__load_dataset()

    def __load_dataset(self):
        print("Loading spacy...")
        try:
            self.nlp = self.spacy.load("en_core_web_sm")
        except OSError:
            print("Model 'en_core_web_sm' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = self.spacy.load("en_core_web_sm")

        print("Downloading required files...\n")
        for ds in ['stopwords', 'wordnet']:
            self.nltk.download(ds)

        print("\nSetting Stopwords...")
        self.stop_words = set(self.stopwords.words('english'))

        print("Initializing Lemmatizer...\n")
        self.lemmatizer = self.WordNetLemmatizer()

    def __do_preprocessing(self, text: str) -> str:
        tokens = []
        text = text.lower().strip()
        doc = self.nlp(text)
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            if token.text in self.stop_words:
                continue
            lemma = self.lemmatizer.lemmatize(token.text)
            tokens.append(lemma)
        return " ".join(tokens)

    def __sentiment(self, text: str) -> float:
        if len(text.split()) < 3:
            return -0.5
        return self.TextBlob(text).sentiment.polarity

    @staticmethod
    def __sentiment_bucket(p: float) -> str:
        if p < -0.2:
            return "negative"
        elif p > 0.2:
            return "positive"
        return "neutral"

    @staticmethod
    def __load_artifacts_for_corpus(path: str, dtype: DatasetType):
        pairs = []
        if not os.path.exists(path):
            print(f"[WARN] Dataset not found at {path}")
            return pairs

        if dtype == DatasetType.WELLNESS:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                patterns = item.get("patterns", [])
                responses = item.get("responses", [])
                if not responses:
                    continue
                for p in patterns:
                    for resp in responses:
                        pairs.append((p, resp))
            print(f"[INFO] Loaded {len(pairs)} wellness pairs")
            return pairs

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if dtype == DatasetType.NPS:
                    if "utterance" in obj and "response" in obj:
                        pairs.append((obj["utterance"], obj["response"]))
                elif dtype == DatasetType.CORNELL:
                    if "input" in obj and "response" in obj:
                        pairs.append((obj["input"], obj["response"]))
        print(f"[INFO] Loaded {len(pairs)} pairs from {dtype.name}")
        return pairs

    def __build_corpus(self):
        processed = []
        nps = ChatbotEngine.__load_artifacts_for_corpus("data/nps_chat.jsonl", DatasetType.NPS)
        cornell = ChatbotEngine.__load_artifacts_for_corpus("data/cornell_dialogs.jsonl", DatasetType.CORNELL)
        wellness = ChatbotEngine.__load_artifacts_for_corpus("data/custom_wellness.json", DatasetType.WELLNESS)
        wellness_weighted = wellness * 3
        all_pairs = nps + cornell + wellness_weighted

        for i, resp in all_pairs:
            p_inp = self.__do_preprocessing(i)
            p_resp = resp.strip()
            if p_inp and p_resp:
                processed.append((p_inp, p_resp))
        print(f"[INFO] Total processed training pairs: {len(processed)}\n")
        self.corpus = processed

        # Build TF-IDF artifacts for hybrid similarity
        self.raw_inputs = [inp for inp, _ in self.corpus]
        self.raw_responses = [resp for _, resp in self.corpus]
        self.vectorizer = self.TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.raw_inputs)

    def __create_bot(self):
        self.bot = self.ChatBot("WellnessMotivationBot", read_only=False,
            logic_adapters=[
                {
                    "import_path": "chatterbot.logic.BestMatch",
                    "statement_comparison_function": "chatterbot.comparisons.cosine_similarity",
                    "response_selection_method": "chatterbot.response_selection.get_most_frequent_response"
                }
            ], storage_adapter="chatterbot.storage.SQLStorageAdapter",
            database_uri="sqlite:///wellness_chatbot.sqlite3")

    def __train_bot(self):
        trainer = self.ListTrainer(self.bot)
        print("[INFO] Training started...")
        for i, (inp, resp) in enumerate(self.corpus):
            trainer.train([inp, resp])
            if (i + 1) % 1000 == 0:
                print(f"[INFO] Trained {i + 1} pairs")
        print("[INFO] Training complete.")

    @staticmethod
    def __dynamic_threshold(bucket: str) -> float:
        if bucket == "negative":
            return 0.40
        elif bucket == "positive":
            return 0.25
        return 0.30

    @staticmethod
    def __does_it_look_like_movie_dialog(text: str) -> bool:
        t = text.lower()
        return any(p in t for p in AdaptiveControlMessages.MOVIE_STYLE_PHRASES)

    @staticmethod
    def __sentiment_route(bucket: str, bot_reply: str) -> str:
        lower = bot_reply.lower()
        if bucket == "negative":
            if any(word in lower for word in ["great", "awesome", "amazing"]):
                return random.choice(AdaptiveControlMessages.FALLBACK_NEG)
        if bucket == "positive":
            if any(word in lower for word in ["sorry", "tough", "hard"]):
                return random.choice(AdaptiveControlMessages.FALLBACK_POS)
        return bot_reply

    def __tfidf_best_match(self, processed_input: str):
        if self.vectorizer is None or self.tfidf_matrix is None:
            return 0.0, None
        vec = self.vectorizer.transform([processed_input])
        sims = self.cosine_similarity(vec, self.tfidf_matrix).flatten()
        best_idx = sims.argmax()
        best_score = sims[best_idx]
        return best_score, self.raw_responses[best_idx]

    @staticmethod
    def __pick_fallback(pool, last):
        choice = random.choice(pool)
        while choice == last and len(pool) > 1:
            choice = random.choice(pool)
        return choice

    def __chat(self):
        print("\n[INFO] Chatbot ready. Type 'quit' to exit.\n")

        while True:
            user = input("You: ").strip()
            if user.lower() == "quit":
                print("Bot: Take care of yourself today. Goodbye.")
                break

            p_user = self.__do_preprocessing(user)
            pol = self.__sentiment(user)
            bucket = self.__sentiment_bucket(pol)
            conf_threshold = self.__dynamic_threshold(bucket)

            # ChatterBot response
            response = self.bot.get_response(p_user)
            cb_conf = float(getattr(response, "confidence", 0.0))
            cb_reply = str(response)

            # TF-IDF response
            tfidf_score, tfidf_reply = self.__tfidf_best_match(p_user)

            # Choose base reply between ChatterBot and TF-IDF
            if tfidf_reply is not None and tfidf_score > cb_conf:
                base_reply = tfidf_reply
                base_conf = tfidf_score
            else:
                base_reply = cb_reply
                base_conf = cb_conf

            # Movie-dialogue suppression
            if self.__does_it_look_like_movie_dialog(base_reply):
                base_conf = 0.0

            # Hybrid score (simple weighted combo)
            hybrid_score = (cb_conf * 0.6) + (tfidf_score * 0.4)

            if hybrid_score < conf_threshold:
                if bucket == "negative":
                    reply = self.__pick_fallback(AdaptiveControlMessages.FALLBACK_NEG, self.last_fallback)
                elif bucket == "positive":
                    reply = self.__pick_fallback(AdaptiveControlMessages.FALLBACK_POS, self.last_fallback)
                else:
                    reply = self.__pick_fallback(AdaptiveControlMessages.FALLBACK_NEU, self.last_fallback)
                self.last_fallback = reply
            else:
                routed = ChatbotEngine.__sentiment_route(bucket, base_reply)
                reply = routed

            print(f"[DEBUG] sentiment={bucket}, polarity={pol:.3f}, "
                  f"cb_conf={cb_conf:.3f}, tfidf={tfidf_score:.3f}, hybrid={hybrid_score:.3f}, "
                  f"threshold={conf_threshold:.2f}")
            print(f"Bot: {reply}\n")

    def run(self):
       self.__build_corpus()
       self.__create_bot()
       self.__train_bot()
       self.__chat()

class TestCaseRunner:
    @staticmethod
    def run_test():
        chatbot_eng = ChatbotEngine()
        chatbot_eng.run()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['nltk', 'textblob', 'chatterbot', 'spacy', 'scikit-learn']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 8 - Portfolio Project | Option 2 ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()