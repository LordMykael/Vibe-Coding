import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Steam data can be obtained from Kaggle datasets such as "steam-200k" (Steam games with reviews and interactions)
# The dataset is expected to have columns like: 'title', 'genres', 'release_date', 'price', 'average_playtime', 'owners', 'positive_ratings', 'negative_ratings'

def load_steam_data(filepath):
    """
    Load and preprocess Steam data from a CSV file.
    Example datasets: https://www.kaggle.com/datasets/nikdavis/steam-store-games or https://www.kaggle.com/datasets/romainbehar/steam-200k
    """
    df = pd.read_csv(filepath)
    return df

def visualizar_grafico(x, y, data, tipo='scatter', titulo='', xlabel='', ylabel=''):
    """
    Visualiza um gráfico usando seaborn e matplotlib.

    Parâmetros:
    - x: coluna para o eixo x
    - y: coluna para o eixo y
    - data: DataFrame de dados
    - tipo: 'scatter' para scatterplot, 'bar' para gráfico de barras, etc.
    - titulo: título do gráfico
    - xlabel: rótulo do eixo x
    - ylabel: rótulo do eixo y
    """
    plt.figure(figsize=(8, 6))
    if tipo == 'scatter':
        sns.scatterplot(x=x, y=y, data=data)
    elif tipo == 'bar':
        sns.barplot(x=x, y=y, data=data)
    else:
        raise ValueError("Tipo de gráfico não suportado.")
    plt.title(titulo)
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)
    plt.tight_layout()
    plt.show()
def preprocess_steam_data(df):
    # Convert genres to string if not already
    df['genres'] = df['genres'].astype(str)

    # Parse release year from release_date if available
    if 'release_date' in df:
        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    # Create a success metric: e.g., Total ratings or Owners or (positive - negative ratings)
    df['total_ratings'] = df.get('positive_ratings', 0) + df.get('negative_ratings', 0)
    if 'owners' in df:
        # Owners is a string range, take the upper bound
        df['owners_high'] = df['owners'].astype(str).str.extract('(\d+,\d+)$')[0]
        df['owners_high'] = df['owners_high'].str.replace(',', '').astype(float)
    else:
        df['owners_high'] = np.nan

    # You can define 'success' as e.g., high owners, high total positive ratings, or reviews/price, etc.
    df['success_score'] = df.get('positive_ratings', 0) - df.get('negative_ratings', 0)
    return df

def analyze_steam_success_factors(df):
    # What characteristics distinguish more successful games?
    
    # Correlation with owners (if available)
    numeric_features = ['price', 'average_playtime', 'total_ratings', 'positive_ratings', 'negative_ratings', 'success_score', 'owners_high']
    available_features = [f for f in numeric_features if f in df.columns]
    corr = df[available_features].corr()
    print("Correlation with Owners/Positive Ratings:\n", corr.get('owners_high', corr.get('success_score')))

    # Genre analysis
    genre_df = df[df['genres'].notnull() & (df['genres'] != '')]
    genres_exploded = genre_df.assign(genre=genre_df['genres'].str.split(';')).explode('genre')
    if 'owners_high' in genres_exploded:
        sales_by_genre = genres_exploded.groupby('genre')['owners_high'].mean().sort_values(ascending=False)
        print("\nAverage Owners by Genre:\n", sales_by_genre)
    else:
        ratings_by_genre = genres_exploded.groupby('genre')['success_score'].mean().sort_values(ascending=False)
        print("\nAverage Success Score by Genre:\n", ratings_by_genre)

    # Price analysis
    if 'owners_high' in df:
        plt.figure(figsize=(7,5))
        sns.scatterplot(x='price', y='owners_high', data=df)
        plt.title('Game Price vs Owners')
        plt.xlabel('Price (USD)')
        plt.ylabel('Number of Owners (approx)')
        plt.show()
    
    # Playtime analysis
    if 'average_playtime' in df and 'owners_high' in df:
        plt.figure(figsize=(7,5))
        sns.scatterplot(x='average_playtime', y='owners_high', data=df)
        plt.title('Average Playtime vs Owners')
        plt.xlabel('Average Playtime (minutes)')
        plt.ylabel('Number of Owners (approx)')
        plt.show()

    # Release year trends
    if 'release_year' in df and 'owners_high' in df:
        yearly_owners = df.groupby('release_year')['owners_high'].sum()
        yearly_owners.plot(title="Total Owners by Release Year")
        plt.xlabel('Year')
        plt.ylabel('Total Owners (approx)')
        plt.show()

    # Genre boxplot
    if 'owners_high' in genres_exploded:
        plt.figure(figsize=(9,5))
        top_genres = genres_exploded['genre'].value_counts().index[:10]
        sns.boxplot(x='genre', y='owners_high', data=genres_exploded[genres_exploded['genre'].isin(top_genres)])
        plt.title('Owner Distribution by Genre (Steam)')
        plt.xticks(rotation=45)
        plt.show()

    # Success vs ratings
    if 'positive_ratings' in df and 'owners_high' in df:
        plt.figure(figsize=(7,5))
        sns.scatterplot(x='positive_ratings', y='owners_high', data=df)
        plt.title('Positive Ratings vs Owners')
        plt.xlabel('Positive Ratings')
        plt.ylabel('Owners')
        plt.show()

def main():
    # Example usage:
    filepath = 'steam_games.csv'  # Altere para o arquivo da base de dados da Steam
    df = load_steam_data(filepath)
    df = preprocess_steam_data(df)
    analyze_steam_success_factors(df)

# Descomente a linha abaixo para rodar a análise quando executar o script
main()