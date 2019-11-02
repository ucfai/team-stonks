from sklearn.feature_extraction.text import CountVectorizer


def FeatureVectorization(data):
    vectorizer = CountVectorizer(
        analyzer="word",
        lowercase=False,
    )

    features = vectorizer.fit_transform(data)
    return features.toarray()
