import random
import dill
from nltk.corpus import movie_reviews
from textblob.classifiers import NaiveBayesClassifier

def create_sentiment_model():

    random.seed(1)

    # Grab some movie review data
    reviews = [(list(movie_reviews.words(fileid)), category)
                  for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)]
    random.shuffle(reviews)
    new_train, new_test = reviews[:1900], reviews[1900:]

    cl = NaiveBayesClassifier(new_train)

    # Compute accuracy
    accuracy = cl.accuracy(new_test)
    print("Accuracy: {0}".format(accuracy))

    # Show 5 most informative features
    print cl.show_informative_features(5)

    with open('sentiment_clf_full.pkl', 'wb') as pk:
        dill.dump(cl, pk)
    print 'done saving model'

if __name__ == "__main__":
    create_sentiment_model()    