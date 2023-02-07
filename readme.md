# Autoscraper

## Description and usage

This is a good robot if you need to scale up your scraping and you don't want to write 100s of scripts. This is not a good bot if you're having trouble with `requests` or `selenium` or you're at a stage where you can't do QA.

You need to have positive and negative examples from at least one website before using the script, meaning `positive == text I want`, `negative == text I don't want`. 

### Autoscraper
The `Autoscraper()` class can identify the text you want to scrape from any website after it has been trained. You need to instantiate it and then pass training examples and their labels like so:

```
autoscraper = Autoscraper()

positive_labels = [1] * len(text_I_want)
negative_labels = [0] * len(text_I_don't_want)

text = text_I_want + text_I_don't_want
labels = positive_labels + negative_labels

autoscraper.fit(text, labels)

results = autoscraper.predict(text_from_new_website)

```

#### .fit()
The expected arguments are `text` (list), `labels` (list). Optional: `regression`, `"Ridge", "logistic", *object*` (see below). 

#### .predict()
The expected arguments are `text` (list). Returns `0, 1`, meaning `negative, positive`, or `discard, keep`.

### Autonav

The `Autonav()` class creates a path to navigate the website using selenium. Please be advised that the compute scales quadratically. 

#### .predict()
The expected arguments are `pages` (list), `{'text': result}` (dict). You will need the results from Autoscraper first.

## Statistical considerations

The classifier uses the following matrix to make a decision"

```
matrix = [[text_scores, --> n-gram classifier
        dep_scores, --> dependency classifier
        pos_scores, --> part-of-speech classifier
        tag_scores, --> named entity classifier
        shape_scores, --> lower/upper case classifier
        word_counts, --> word count
        character_counts, --> character count
        alphas, --> number of alphanumeric words ((?:\b\d+))
        stopws, --> number of stopwords
        n_digits--> number of digits ([0-9])
        ]
        ...
        ]
```

As you can see, there is a number of redundant features to amplify the signal and widen the decision boundary. However, this also means high multicollinearity and Gauss-Markov does not hold. The default regression method is Ridge from sklearn. If you have a small dataset where bias-variance might be a larger source of error than collinearity, you can pass the argument `regression = "logistic"` in the `.fit()` method or a classifier of your choice as an object.

### Error rate and case study
My subjective recollection is that the error rate was around 10% when scraping newspapers for [articles, authors, metrics, comments]

## License

Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)

https://creativecommons.org/licenses/by-nc/3.0/
