import pandas as pd
import sqlite3
from vanna.remote import VannaDefault
import vanna as vn


print("\nLoading CSV data...")
df = pd.read_csv(r'C:\ufs\tasks\processed_cleveland.csv')

# Create SQLite connection
conn = sqlite3.connect('data.db')
df.to_sql('cleveland_data', conn, if_exists='replace', index=False)
conn.close()

print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
print(f"✓ Columns: {list(df.columns)}\n")

# IMPORTANT: Create your model at https://vanna.ai/account/profile first!
# Then use that model name here
model_name = 'cleveland-heart-model'  # Your model name from Vanna.ai
print(f"Connecting to model: {model_name}...")
print("(If this fails, create the model at https://vanna.ai/account/profile first)\n")

vn_client = VannaDefault(model="artman-jr", api_key="07d5f470733f4d34ba47420daf195b10")
vn_client.connect_to_sqlite('data.db')
print("✓ Connected to database\n")

print("✓ Connected to database\n")

print("Training model with example queries...")
try:
    # First, let's add some documentation about the table
    vn_client.train(
        ddl="""
        CREATE TABLE cleveland_data (
            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            num INTEGER
        )
        """
    )
    
    # Add documentation about the columns
    vn_client.train(
        documentation="The cleveland_data table contains heart disease data. "
        "num column indicates presence of heart disease (0=no disease, 1-4=disease present)."
    )
    
    # Train with example question-SQL pairs
    vn_client.train(
        question="How many rows are in cleveland_data?",
        sql="SELECT COUNT(*) FROM cleveland_data"
    )
    vn_client.train(
        question="What is the average age?",
        sql="SELECT AVG(age) FROM cleveland_data"
    )
    
    print("✓ Training complete\n")
except Exception as e:
    print(f"⚠ Training skipped (model may need initialization): {str(e)}\n")
    print("  You can still ask questions - the model will do its best!\n")

print("="*70)
print("INTERACTIVE QUESTION MODE")
print("="*70)
print("You can now ask questions about your data!")
print("Type 'exit' or 'quit' to stop")
print("\nTable name: cleveland_data")
print(f"Columns: {', '.join(df.columns.tolist())}")
print("\nExample questions:")
print("  - How many rows are in cleveland_data?")
print("  - What is the average age in cleveland_data?")
print("  - Show me the first 10 rows from cleveland_data")
print("  - What is the distribution of heart disease by sex?")
print("="*70 + "\n")

while True:
    # Get user question
    question = input("Your question: ").strip()
    
    # Check if user wants to exit
    if question.lower() in ['exit', 'quit', 'q', '']:
        print("\nGoodbye! 👋")
        break
    
    try:
        # Ask Vanna
        print("\n🤔 Thinking...")
        response = vn_client.ask(question)
        print(f"\n💡 Answer: {response}\n")
        print("-"*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        print("Try rephrasing your question.\n")
        print("-"*70 + "\n")