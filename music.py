import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import simpledialog, ttk
from sklearn.metrics.pairwise import cosine_similarity

# Load data from CSV files
user_df = pd.read_csv('user.csv')
songs_df = pd.read_csv('songs.csv')
history_df = pd.read_csv('listen_history.csv')

# Create a pivot table for user-song listen counts
pivot_table = history_df.pivot(index='user_id', columns='song_id', values='listen_count').fillna(0)

# Compute song similarity matrix using cosine similarity
song_similarity = cosine_similarity(pivot_table.T)
song_similarity_df = pd.DataFrame(song_similarity, index=pivot_table.columns, columns=pivot_table.columns)

def hybrid_recommend_songs(user_id, pivot_table, song_similarity_df, songs_df, num_recommendations=5):
    """
    Recommend songs for a given user based on a hybrid approach of user-based and item-based collaborative filtering.
    
    Parameters:
    user_id (int): ID of the user for whom to recommend songs.
    pivot_table (pd.DataFrame): User-song listen count matrix.
    song_similarity_df (pd.DataFrame): Song similarity matrix.
    songs_df (pd.DataFrame): DataFrame containing song information.
    num_recommendations (int): Number of song recommendations to return.
    
    Returns:
    pd.DataFrame: DataFrame containing recommended songs.
    """
    try:
        # Get the user's listening history
        user_vector = pivot_table.loc[user_id]

        # Compute user-based scores
        user_similarity = pivot_table.dot(user_vector) / (pivot_table.sum(axis=1) * user_vector.sum()).clip(lower=1e-9)
        user_similarity = user_similarity.drop(user_id)  # Drop self-similarity
        similar_users = user_similarity.sort_values(ascending=False).index
        similar_users_data = pivot_table.loc[similar_users]
        user_based_scores = similar_users_data.sum().sort_values(ascending=False)

        # Compute item-based scores
        listened_songs = user_vector[user_vector > 0].index
        item_based_scores = song_similarity_df.loc[listened_songs].sum().sort_values(ascending=False)

        # Combine scores
        combined_scores = user_based_scores + item_based_scores

        # Filter out songs the user has already listened to
        already_listened = user_vector[user_vector > 0].index
        recommendations = combined_scores.drop(already_listened)
        
        # Sort recommendations by user's listen count preference
        recommendations = recommendations.sort_values(ascending=False)
        
        # Return top songs based on user's preference
        top_songs = recommendations.head(num_recommendations)
        
        # Return recommended songs
        return songs_df[songs_df['song_id'].isin(top_songs.index)]

    except KeyError:
        print(f"User {user_id} not found in the data.")
        return pd.DataFrame()

def display_recommendations(recommendations):
    """
    Display the recommended songs in a Tkinter window.
    
    Parameters:
    recommendations (pd.DataFrame): DataFrame containing recommended songs.
    """
    if recommendations.empty:
        print("No recommendations found.")
        return
    
    root = tk.Tk()
    root.title("Recommended Songs")
    
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Add the Treeview widget
    tree = ttk.Treeview(frame, columns=("Song ID", "Song Name", "Artist"), show="headings")
    tree.heading("Song ID", text="Song ID")
    tree.heading("Song Name", text="Song Name")
    tree.heading("Artist", text="Artist")
    
    # Insert recommended song data into the Treeview
    for _, row in recommendations.iterrows():
        tree.insert("", tk.END, values=(row["song_id"], row["title"], row["artist"]))
    
    tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    root.mainloop()

def display_user_preferences(user_id, pivot_table, songs_df):
    """
    Display the user's top preferences based on maximum listen counts.
    
    Parameters:
    user_id (int): ID of the user.
    pivot_table (pd.DataFrame): User-song listen count matrix.
    songs_df (pd.DataFrame): DataFrame containing song information.
    """
    root = tk.Tk()
    root.title("User Preferences")
    
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    try:
        user_vector = pivot_table.loc[user_id]
        top_songs = user_vector.sort_values(ascending=False).head(5)
        preferences = songs_df[songs_df['song_id'].isin(top_songs.index)]
        
        # Add the Treeview widget
        tree = ttk.Treeview(frame, columns=("Song ID", "Song Name", "Artist"), show="headings")
        tree.heading("Song ID", text="Song ID")
        tree.heading("Song Name", text="Song Name")
        tree.heading("Artist", text="Artist")
        
        # Insert song data into the Treeview
        for _, row in preferences.iterrows():
            tree.insert("", tk.END, values=(row["song_id"], row["title"], row["artist"]))
        
        tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    except KeyError:
        # Display message if user ID is not found
        label = ttk.Label(frame, text=f"User {user_id} not found in the data.", font=("Helvetica", 12))
        label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    
    root.mainloop()

# Ask the user to input their user ID
root = tk.Tk()
root.withdraw()  # Hide the root window
user_id = simpledialog.askinteger("Input", "Please enter your user ID:", parent=root)

if user_id is not None:
    display_user_preferences(user_id, pivot_table, songs_df)
    recommendations = hybrid_recommend_songs(user_id, pivot_table, song_similarity_df, songs_df)
    display_recommendations(recommendations)
else:
    print("User ID input was cancelled.")
