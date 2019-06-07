from flask import Flask, jsonify
import pandas as pd
import numpy as np
import textdistance

app = Flask(__name__)

df = pd.read_csv("precomputed_neighbors.csv", index_col=0)

# compute Jaro-Winkler distance calculation
JW = JW = textdistance.JaroWinkler(long_tolerance=True)

# vectorize the function for increase speed
vec_JW = np.vectorize(lambda a, b: JW.normalized_similarity(a, b))

def get_recommendations(movie_id):
    """
    Return 5 movies among the nearest 20. The final 5 at picked at random,
    based on their score (when the give movie's score > 5), and the squared
    euclidian distance.
    """

    if movie_id not in df.index:
        return {'error': 'The ID number is unknown'}

    series = df.loc[movie_id]

    title = series.movie_title
    score = series.imdb_score

    def s_split(string):
        idx, title, score, dist = string.split("|")
        return [int(idx), title, float(score), float(dist)]

    # first two columns are movie_title and imdb_score which are
    # not realted to the neighbors
    values = series[2:].apply(s_split).tolist()

    result_df = pd.DataFrame(
        data=values,
        columns=['id', 'title', 'imdb_score', 'distance']
    )

    # take score into consideration only if the score of the base movie is over 5
    if score >= 5:
        result_df['p'] = result_df['imdb_score'] / result_df['distance'] ** 2
    # otherwise it is not meaningful
    else:
        result_df['p'] = 1 / result_df['distance'] ** 2

    # normalize probability values such as sum(p) = 1
    result_df['p'] = result_df['p'] / result_df['p'].sum()

    # pick 5 different movies at random based on p
    random_index = np.random.choice(result_df.index, size=5, replace=False, p=result_df.p)
    result_dict = result_df.loc[random_index].drop(columns=['distance', 'p']).to_dict(orient='index')

    # order results by distance
    result_list = [result_dict[key] for key in sorted(result_dict.keys())]

    return {
        'id': movie_id,
        'title': title,
        'imdb_score': score,
        'results': result_list,
    }

def most_similar_titles(n, title, jw_dist_thresh=0.8):
    """
    Return index of the movie titles matching a string.
    Maximum is n results, with a score over the thresh.
    """
    distances = pd.Series(vec_JW(title.lower(), df.movie_title.str.lower()),
                          index=df.index)
    return distances[distances > jw_dist_thresh].nlargest(n).index

@app.route('/recommend/<movie_id>', methods=['GET'])
def recommend(movie_id):

    try:
        movie_id = int(movie_id)
    except:
        return jsonify({'error': 'The ID must be an integer'})

    results = get_recommendations(movie_id)

    return jsonify(results)

@app.route('/recommend/fuzzy/<string>', methods=['GET'])
def fuzzy(string):

    index = most_similar_titles(5, str(string))
    if index:
        results = {
            "fuzzy title": string,
            "matched": [get_recommendations(i) for i in index]
        }
    else:
        results = {'error': 'No matching movie title'}


    return jsonify(results)

if __name__ == '__main__':
    # do not sort keys in alphabetical order
    app.config["JSON_SORT_KEYS"] = False
    app.run(debug=True)