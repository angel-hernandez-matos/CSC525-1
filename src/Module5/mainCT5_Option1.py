# File: mainCT5.py
# Written by: Angel Hernandez
# Description: Module 5 - Critical Thinking - Option 1
# Requirement(s): Text Dataset Augmentation
# For your assignment, submit a Python script that will take any text dataset and augment it in some way to
# expand the dataset. Submission must include a script that will augment any text dataset within its folder. Please
# include the un-augmented dataset with the augmented dataset and a short description of what was augmented.

import os
import sys
import subprocess
import random

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class TextAugmenter:
    def __init__(self):
       import nltk as nltk
       from nltk.corpus import wordnet as wn
       from datasets import load_dataset as ld
       self.nltk = nltk
       self.wordnet = wn
       self.dataset = None
       self.load_dataset = ld
       self.original_folder = "original"
       self.augmented_folder = "augmented"
       self.original_path = os.path.join(self.original_folder, "sst2_original.txt")
       self.augmented_path = os.path.join(self.augmented_folder, "sst2_augmented.txt")
       for f in [self.original_folder, self.augmented_folder]: os.makedirs(f, exist_ok=True)
       self.__load_dataset()

    def __load_dataset(self):
        print("Downloading required files...\n")
        for ds in ['wordnet', 'omw-1.4']: self.nltk.download(ds)
        print("\nLoading SST-2 dataset...")
        self.dataset = self.load_dataset("glue", "sst2")["train"]

    def __get_synonym(self, word):
        synsets = self.wordnet.synsets(word)
        if not synsets:
            return word
        lemmas = synsets[0].lemmas()
        if not lemmas:
            return word
        synonym = lemmas[0].name().replace("_", " ")
        return synonym if synonym.lower() != word.lower() else word

    def __synonym_replacement(self, sentence, x=2):
        new_words = []
        words = sentence.split()
        if len(words) == 0:
            return sentence
        words_to_replace = random.sample(words, min(x, len(words)))
        for w in words:
            if w in words_to_replace:
                new_words.append(self.__get_synonym(w))
            else:
                new_words.append(w)
        return " ".join(new_words)

    @staticmethod
    def __random_deletion(sentence, prob=0.1):
        words = sentence.split()
        if len(words) == 0:
            return sentence
        new_words = [w for w in words if random.random() > prob]
        # Let's make sure that at least one word remains
        return " ".join(new_words) if new_words else random.choice(words)

    def __augment_text(self, sentence):
        augmented1 = self.__synonym_replacement(sentence)
        augmented2 = self.__random_deletion(sentence)
        return [augmented1, augmented2]

    def do_augmentation(self):
        with open(self.original_path, "w", encoding="utf-8") as orig_f, \
                open(self.augmented_path, "w", encoding="utf-8") as aug_f:
            for item in self.dataset:
                sentence = item["sentence"]
                label = item["label"]
                orig_f.write(f"{label}\t{sentence}\n")
                # Generate augmented versions
                augmented = self.__augment_text(sentence)
                for aug in augmented:
                    aug_f.write(f"{label}\t{aug}\n")

        print("Augmentation complete.")
        print("Original dataset saved to:", self.original_path)
        print("Augmented dataset saved to:", self.augmented_path)
        print()

class TestCaseRunner:
    @staticmethod
    def run_test():
        text_augmenter = TextAugmenter()
        text_augmenter.do_augmentation()

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['nltk', 'datasets']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 5 - Critical Thinking | Option 1 ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()