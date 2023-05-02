# conda environment py36 ONLY
import turicreate as tc

PATH = 'manning/Chapter_6_Logistic_Regression/IMDB_Dataset.csv'


def get_frame(path2csv):
    movies = tc.SFrame(path2csv)
    movies['words'] = tc.text_analytics.count_words(movies['review'])
    return movies

def model(sframe, target, *features):
    return tc.logistic_classifier.create(sframe, features=list(features), target=target)


if __name__ == '__main__':

    movies = get_frame(PATH)
    md = model(movies, 'sentiment', 'words')

    movies['predictions'] = md.predict(movies, output_type='probability') 
    print(movies)