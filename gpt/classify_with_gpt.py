import pandas as pd
import openai
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI()

def process_file(input_file, output_dir, instructions):
    """Process a single test file and save results"""
    print(f"\nProcessing {input_file}...")
    
    # Load test data
    test_df = pd.read_csv(input_file)
    total_rows = len(test_df)
    
    # Initialize results storage
    results_data = []
    
    # Calculate estimated completion time
    estimated_time = total_rows * 0.2  # 0.2 seconds per request
    completion_time = datetime.now() + timedelta(seconds=estimated_time)
    print(f"Estimated completion time: {completion_time.strftime('%H:%M:%S')}")
    
    # Process test data
    for idx, row in tqdm(test_df.iterrows(), total=total_rows):
        text = row['text']
        true_label = 0 if row['label'] == "democratic" else 1
        
        # Get prediction
        try:
            response = client.responses.create(
                model="gpt-4.1-nano-2025-04-14",
                instructions=instructions,
                input=text,
                temperature=0,  # Lower temperature for more consistent outputs
            )
            predicted_label = response.output_text
            
        except Exception as e:
            print(f"Error in API call for row {idx}: {e}")
            predicted_label = None
        
        # Store results
        results_data.append({
            'text': text,
            'true_label': true_label,
            'predicted_label': predicted_label,
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results_data)
    
    # Get file suffix and create output filename
    file_suffix = os.path.basename(input_file).split('_')[-1].split('.')[0]
    output_file = os.path.join(output_dir, f'results_{file_suffix}.csv')
    
    # Save results to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

def main():
    # Create output directory
    output_dir = 'classification_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load instructions
    with open("prompt.txt", "r", encoding="utf-8") as f:
        instructions = f.read()
    
    test_dir = 'split_test_data'
    for i in range(1, 111):  # Process files 1 through 10
        input_file = f'test_part_{i}.csv'
        input_path = os.path.join(test_dir, input_file)
        output_file = os.path.join(output_dir, f'results_{i}.csv')
        
        # Skip if result file already exists
        if os.path.exists(output_file):
            print(f"\nSkipping {input_file} - results already exist in {output_file}")
            continue

        if os.path.exists(input_path):
            process_file(input_path, output_dir, instructions)
        else:
            print(f"Warning: {input_file} not found")

if __name__ == "__main__":
    main()