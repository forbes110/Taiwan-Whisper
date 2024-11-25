import pandas as pd
import argparse

def process_csv(csv_file, output_file):
    """
    Check for duplicate rows in a CSV file, count distinct duplicates, 
    and generate a version without duplicates.
    """
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(csv_file)

        # Check for duplicates
        duplicates = data[data.duplicated()]  # Get all duplicate rows
        if duplicates.empty:
            print("No duplicate rows found.")
        else:
            # Count the number of distinct duplicate rows
            distinct_duplicates = duplicates.drop_duplicates()
            print(f"Duplicate rows found:\n{duplicates}")
            print(f"Total duplicate rows: {len(duplicates)}")
            print(f"Number of distinct duplicate rows: {len(distinct_duplicates)}")
            print(f"Distinct duplicate rows:\n{distinct_duplicates}")

        # Create a version of the DataFrame without duplicates
        data_no_duplicates = data.drop_duplicates()

        # Save the version without duplicates to a new CSV file
        data_no_duplicates.to_csv(output_file, index=False)
        print(f"CSV file without duplicates saved to: {output_file}")

    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Check for duplicate rows, count distinct duplicates, and remove duplicates.")
    parser.add_argument("--csv_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, default="done_channel_names_.csv", help="Path to the output CSV file without duplicates.")
    args = parser.parse_args()

    # Process the CSV file
    process_csv(args.csv_file, args.output_file)
