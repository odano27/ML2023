import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import concurrent.futures

#import
users = pd.read_csv('users_cleaned.csv', delimiter=',', low_memory=False)
df1 = pd.read_csv('animelists_cleaned (1).txt', sep=',', low_memory=False)
df2 = pd.read_csv('animelists_cleaned (2).txt', sep=',', low_memory=False)
df3 = pd.read_csv('animelists_cleaned (3).txt', sep=',', low_memory=False)
#df4 = pd.read_csv('animelists_cleaned (4).txt', sep=',', low_memory=False)
#df5 = pd.read_csv('animelists_cleaned (5).txt', sep=',', low_memory=False)
anime_f = pd.read_csv('anime_cleaned.txt', delimiter=',')
anime = anime_f[['anime_id','title', 'type', 'airing', 'duration',
       'rating', 'score', 'scored_by', 'rank', 'popularity', 'members',
       'favorites', 'studio', 'genre', 'duration_min', 'aired_from_year']].dropna(subset=['rating','rank','genre'])
#Anime music is irrelevant
anime = anime[anime['type'] != 'Music']
#take only animes watched and scored by more than 100 people
anime = anime[anime['scored_by']>100]
#filtering users who scored more than 20 animes
animelists = pd.concat([df1,df2,df3], axis=0)
animelists = animelists.merge(users[['username', 'user_id']], 
                              on='username', how='left')
animelists=animelists[animelists['my_score'] > 0]
watched_count=animelists['user_id'].value_counts()
watched_more_than_20=watched_count[watched_count > 20].index
#watched_more_than_20=pd.Series(watched_more_than_20).sample(n=5000, random_state=42).tolist()
animelists_filtered=animelists[animelists['user_id'].
                      isin(watched_more_than_20)]
#create a user/anime score table based on the filtered list
user_anime_score_table = animelists_filtered.pivot(index='user_id',
                                          columns='anime_id',
                                          values='my_score')
user_anime_score_table = user_anime_score_table.loc[:,
                         user_anime_score_table.columns.isin(
                            anime['anime_id'])]
#normalise the scores around for the similarity tables in kNN
avg_score = user_anime_score_table.mean(axis=1)
user_anime_score_normalised = user_anime_score_table.sub(avg_score, axis=0).fillna(0)

#%%time K-NN

n = 100
predictions = []

np.random.seed(42)
for _ in range(n):
    #randomly select a user_id
    valid_choice = False
    
    while not valid_choice:
        #randomly select a user_id
        user_id = np.random.choice(user_anime_score_table.index)

        #choose an anime that the user has watched
        watched_anime = user_anime_score_table.loc[user_id].dropna()
        if len(watched_anime) >= 10:
            valid_choice = True
    n_animes_for_user = 10 
    selected_anime_ids = np.random.choice(watched_anime.index, n_animes_for_user, replace=False)
    for anime_id in selected_anime_ids:
        if user_anime_score_table[anime_id].count() > 50:


            # Copy dataframes
            temp_normalized = user_anime_score_normalised.copy()
            temp_original = user_anime_score_table.copy()

            temp_normalized.drop(anime_id, axis=1, inplace=True)

            #user predicting
            target_user_x = temp_normalized.loc[[user_id]]

            #target data from user_anime_score_table
            other_users_y = temp_original[anime_id]

            #data for only those that have seen the anime
            other_users_x = temp_normalized[other_users_y.notnull()]

            #remove those that have not seen the anime from the target
            other_users_y.dropna(inplace=True)

            # kNN
            knn = KNeighborsRegressor(metric='cosine', n_neighbors=10)
            knn.fit(other_users_x, other_users_y)
            user_user_pred = knn.predict(target_user_x)

            #store the user_id, anime_id and predicted score
            predictions.append((user_id, anime_id, user_user_pred[0]))

#convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['user_id', 'anime_id', 'predicted_score'])
predictions_df['actual_score'] = [
    user_anime_score_table.at[user_id, anime_id]
    for user_id, anime_id in zip(predictions_df['user_id'], predictions_df['anime_id'])
]
print(predictions_df)
predictions_df.to_csv('predictions_kNN_user.csv', index=False)

#joining the datasets
df = animelists.iloc[:, [1, 5, 11]].copy()
df.columns = ['anime_id', 'my_score', 'user_id']
selected_user_columns = users.loc[:, ['user_id', 'stats_mean_score', 'birth_date', 'location', 'gender']]
df = pd.merge(df, selected_user_columns, on='user_id', how='left')
df = pd.merge(df, anime, on='anime_id', how='left')
df = df.drop(columns=['title'])
#change DOB to age, city to country
df['birth_year'] = df['birth_date'].str[:4].astype(int)
df['age'] = 2023 - df['birth_year']
df = df.drop(columns=['birth_date', 'birth_year'])
df = df.dropna(subset=['location','genre'])
df.loc[:, 'country'] = df.copy()['location'].str.split(',').str[-1].str.strip()
df = df.drop(columns=['location'])
#encode
categorical_columns = ['gender', 'type', 'airing', 'duration', 'rating', 'studio', 'country']
for col in categorical_columns:
    df[col] = pd.Categorical(df[col]).codes
#one-hot encode genres
all_genres = set()
for genres in df['genre'].str.replace('\s+', '', regex=True).dropna().str.split(','):
    all_genres.update(genres)
genre_encoding = pd.DataFrame(index=df.index)
for genre in all_genres:
    genre = genre.strip()
    genre_encoding[genre] = df['genre'].str.contains(genre).astype(int)
df = pd.concat([df, genre_encoding], axis=1)
df = df.drop(columns=[genre], axis=1)

# RFR

user_ids = df['user_id'].unique()

results_df = pd.DataFrame(columns=['user_id', 'anime_id', 'actual_score', 'predicted_score', 'val_mae', 'val_mse'])

n = 100

# Randomly select n users
chosen_user_ids = np.random.choice(user_ids, size=n, replace=False)

# Model training and prediction for each chosen user
for chosen_user_id in chosen_user_ids:
    
    user_df = df[df['user_id'] == chosen_user_id]
    if len(user_df) > 10:
        X = user_df.drop(columns=['my_score', 'user_id', 'genre'])  
        y = user_df['my_score']

        # Split data into training+validation and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Further split training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111111111111111111, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Fit the model using training data
        model.fit(X_train, y_train)

        # Predict and evaluate using validation data
        val_predictions = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_mse = mean_squared_error(y_val, val_predictions)

        # Predict and evaluate using test data
        test_predictions = model.predict(X_test)

        # Save the results to results_df
        temp_results = pd.DataFrame({
            'user_id': chosen_user_id,
            'anime_id': X_test['anime_id'].values,
            'actual_score': y_test.values,
            'predicted_score': test_predictions,
            'val_mae': val_mae,
            'val_mse': val_mse,
        })

        results_df = pd.concat([results_df, temp_results], ignore_index=True)
        
results_df.to_csv('predictions_rfr.csv', index = False)

#visualise

actual_scores = predictions_df['actual_score'].values
predicted_scores = predictions_df['predicted_score'].values

below_6 = predictions_df[predictions_df['predicted_score'] < 6]
above_6 = predictions_df[predictions_df['predicted_score'] >= 6]

def calculate_errors(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

mae_knn,mse_knn,rmse_knn = calculate_errors(actual_scores, predicted_scores)
mae_below, mse_below, rmse_below = calculate_errors(below_6['predicted_score'], below_6['actual_score'])
mae_above, mse_above, rmse_above = calculate_errors(above_6['predicted_score'], above_6['actual_score'])


print(f'k-NN:\nAverage Test MAE: {mae_knn:.4f}')
print(f'Average Test MSE: {mse_knn:.4f}')
print(f'Average Test RMSE: {rmse_knn:.4f}\n')

print(f"For scores below 6:")
print(f"MAE: {mae_below}")
print(f"MSE: {mse_below}")
print(f"RMSE: {rmse_below}\n")

print(f"For scores above or equal to 6:")
print(f"MAE: {mae_above}")
print(f"MSE: {mse_above}")
print(f"RMSE: {rmse_above}\n")

actual_scores = results_df['actual_score'].values
predicted_scores = results_df['predicted_score'].values

below_6 = results_df[results_df['predicted_score'] < 6]
above_6 = results_df[results_df['predicted_score'] >= 6]


mae_rfr,mse_rfr,rmse_rfr = calculate_errors(actual_scores, predicted_scores)
mae_below, mse_below, rmse_below = calculate_errors(below_6['predicted_score'], below_6['actual_score'])
mae_above, mse_above, rmse_above = calculate_errors(above_6['predicted_score'], above_6['actual_score'])


print(f'Random Forest Regressor:\nAverage Test MAE: {mae_rfr:.4f}')
print(f'Average Test MSE: {mse_rfr:.4f}')
print(f'Average Test RMSE: {rmse_rfr:.4f}\n')

print(f"For scores below 6:")
print(f"MAE: {mae_below}")
print(f"MSE: {mse_below}")
print(f"RMSE: {rmse_below}\n")

print(f"For scores above or equal to 6:")
print(f"MAE: {mae_above}")
print(f"MSE: {mse_above}")
print(f"RMSE: {rmse_above}\n")

import seaborn as sns

from tabulate import tabulate
mean_val_mae = results_df['val_mae'].mean()
mean_val_mse = results_df['val_mse'].mean()

comparison_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE'],
    'Validation': [mean_val_mae, mean_val_mse],
    'Test': [mae_rfr, mse_rfr]
})

print(tabulate(comparison_df, headers='keys', tablefmt='pipe', showindex=False))

error_metrics = ['MAE', 'MSE', 'RMSE']
knn_errors = [mae_knn, mse_knn, rmse_knn]
rfr_errors = [mae_rfr, mse_rfr, rmse_rfr]

barWidth = 0.3

r1 = np.arange(len(knn_errors))
r2 = [x + barWidth for x in r1]

plt.bar(r1, knn_errors, width=barWidth, label='k-NN')
plt.bar(r2, rfr_errors, width=barWidth, label='Random Forest Regressor')

plt.xticks([r + barWidth/2 for r in range(len(knn_errors))], error_metrics)
plt.ylabel('Error Value')
plt.title('Error Metrics Comparison between k-NN and Random Forest Regressor')
plt.legend()

plt.show()

results_df['actual_score'] = results_df['actual_score'].astype(int)

plt.figure(figsize=(12,6))
sns.violinplot(x='actual_score', y='predicted_score', data=results_df)
plt.title('Distribution of Predicted Scores for Each Actual Score - RFR')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.show()

predictions_df['actual_score'] = predictions_df['actual_score'].astype(int)

plt.figure(figsize=(12,6))
sns.violinplot(x='actual_score', y='predicted_score', data=predictions_df)
plt.title('Distribution of Predicted Scores for Each Actual Score - k-NN')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.show()

