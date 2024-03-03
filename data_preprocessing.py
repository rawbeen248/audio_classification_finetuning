import pandas as pd
import os
import shutil

def preprocess_data(audio_dir, csv_file, categories, output_dir):
    # Load CSV file
    df = pd.read_csv(csv_file)

    # Filter based on categories
    filtered_df = df[df['category'].isin(categories)]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Copy the filtered audio files to the new directory
    for _, row in filtered_df.iterrows():
        source_path = os.path.join(audio_dir, row['filename'])
        destination_path = os.path.join(output_dir, row['filename'])
        shutil.copy2(source_path, destination_path)

    print(f"Copied {len(filtered_df)} animal sound files to {output_dir}")

    # Save the new dataframe to a new CSV file in the new directory
    filtered_df[['filename', 'category']].to_csv(os.path.join(output_dir, 'animals_sounds.csv'), index=False)
    print(f"Saved filtered CSV to {os.path.join(output_dir, 'animals_meta.csv')}")

if __name__ == "__main__":
    all_sounds = 'environmental_sounds'
    categories = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow']
    preprocess_data(all_sounds, 'sounds_labels.csv', categories, 'animals_sounds')
