from model import load_model
from utils import read_csv, write_csv, clean_text
from inference import InsightsGenerator


def process_file(input_path, output_path):
    # 1. Load model
    model, tokenizer = load_model()

    # 2. Initialize pipeline
    generator = InsightsGenerator(model, tokenizer)

    # 3. Process data
    df = read_csv(input_path)
    df['Problemas de Saúde'] = df['Problemas de Saúde'].apply(clean_text)
    df['insights'] = df['Problemas de Saúde'].apply(generator.generate_insights)

    # 4. Save results
    write_csv(df, output_path)


if __name__ == "__main__":
    process_file('input.csv', 'output.csv')